"""Native worker RPC bridge: wire-v1 records -> the real ``LaneDispatcher``.

The generated ``flash_s5_*`` methods reach :func:`executor_hooks.worker_rpc`.
Legacy ``async_handoff`` messages still use the old ``WorkerHandoffServer``
protocol.  A request carrying a wire-v1 type is routed here instead, and only
here is the real dispatcher constructed by ``s5_startup.install_native_lane``
used.  There is no second dispatcher, no generic RPC fallback and no path that
turns an RPC reply into a completion proof.

Adapter responsibilities (all CPU-visible):

* ``open`` converts the immutable :class:`wire_v1.AdmissionRecord` to exactly
  one local :class:`lane_chunk.ChunkAdmission`; the request's real transaction,
  state epoch and LP lane identity come from the worker's ``LaneRuntime``.
* ``submit`` is one bounded command. A non-target lane returns ``skipped``;
  ``E3Pending`` returns ``pending`` with the same ``op_seq`` unspent.
* ``poll`` builds a :class:`wire_v1.WorkerCompletionReport` only from
  ``LaneDispatcher.poll()`` (host-queried CUDA events). Proofs are cached per
  ``(op_seq, kind)`` so duplicate reports never rewrite a completion, and the
  report watermark is monotone.
* ``bind``/``receipt`` carry the engine's explicit peer-lane/route bind and its
  validated PP consumer receipts to the sender's channel.
* ``release`` drains, releases lane storage, removes the request row and only
  then returns a closed :class:`wire_v1.ReleaseAck`.
* ``cancel`` is one bounded tick: abort, one host query, and release only when
  the dispatcher actually drained. Otherwise every owner stays on the worker.
"""
import os
import time

from async_handoff import OpenPrefill
from e3_types import (
    CompletionKind,
    CompletionProof,
    DependencyRole,
    E3ContractError,
    E3Pending,
    OperationTicket,
)
from lane_chunk import ChunkAdmission
from lane_state import BatchRow, MetadataSnapshot
from graph_router import TransitionAuthorization
from wire_v1 import (
    BootstrapAck,
    BootstrapCommand,
    CancelAck,
    CancelRequest,
    CompletionCursor,
    OpenAck,
    OpenRequest,
    PPReceiptAck,
    PPReceiptDelivery,
    PeerBinding,
    PeerBindingAck,
    PollRequest,
    QsaProfileQuery,
    QsaProfileObservation,
    ReleaseAck,
    ReleaseRequest,
    ROUTE_BANK_FULL,
    ROUTE_BANK_SPLIT,
    SegmentAcceptance,
    SegmentCommand,
    SegmentKind,
    SourceLeaseDescriptor,
    TransitionAck,
    TransitionCommand,
    TransitionPending,
    WorkerCompletionReport,
    LEAN_DIGEST_START,
    LeanCompletionReport,
    lean_digest,
)


class WorkerBridgeError(E3ContractError):
    pass


class _LeanSent:
    """Held-g L4: what this rank already sent in its lean transaction."""
    __slots__ = ('completed', 'proof_keys', 'hosts', 'due', 'digest', 'last')

    def __init__(self):
        self.completed = set()
        self.proof_keys = set()
        self.hosts = set()
        # Sent completions whose STATE_WRITE / PP_CONSUMER proof is still owed.
        self.due = set()
        self.digest = LEAN_DIGEST_START
        # (pending, watermark, cursor, failed_code) of the last sent report.
        self.last = None

    def end(self):
        return len(self.completed), len(self.proof_keys), len(self.hosts), self.digest


_SEGMENT_ROLE = {
    SegmentKind.PREPARE: DependencyRole.PRODUCER,
    SegmentKind.PREFIX: DependencyRole.PRODUCER,
    SegmentKind.LAYER: DependencyRole.PRODUCER,
    SegmentKind.SUFFIX: DependencyRole.PRODUCER,
    SegmentKind.STATE_FINALIZE: DependencyRole.COMMIT,
    SegmentKind.PP_SEND: DependencyRole.HANDOFF,
    SegmentKind.PP_RECEIVE: DependencyRole.HANDOFF,
}


_MEMORY_FIELDS = (('allocated', 'allocated_bytes.all.current'),
                  ('reserved', 'reserved_bytes.all.current'),
                  ('inactive_split', 'inactive_split_bytes.all.current'),
                  ('alloc_retries', 'num_alloc_retries'),
                  ('ooms', 'num_ooms'),
                  # M3: allocator pressure as a source of device-wide synchronization.
                  ('sync_all_streams', 'num_sync_all_streams'),
                  ('device_allocs', 'num_device_alloc'),
                  ('device_frees', 'num_device_free'))


def memory_receipt(point, rank, transaction, cuda=None):
    """One rank's caching-allocator counters and driver free/total bytes.

    Receipt only (auto-arm runs): host queries of the current device, never a
    CUDA initialization, a synchronization or a control-flow input. Returns
    ``None`` when CUDA is not initialized in this process.
    """
    if cuda is None:
        import torch
        cuda = torch.cuda
    if not cuda.is_initialized():
        return None
    stats = cuda.memory_stats()
    free, total = cuda.mem_get_info()
    row = dict(point=point, rank=int(rank), txn_seq=int(transaction.txn_seq),
               chunk_start=int(transaction.chunk_start), chunk_end=int(transaction.chunk_end))
    for name, key in _MEMORY_FIELDS:
        row[name] = int(stats.get(key, -1))
    row['free'], row['total'] = int(free), int(total)
    return row


PEAK_LINE = 'flash-s5 memory-peak '


def peak_reset(cuda=None):
    """alpha-next: start a chunk's peak window (host-side allocator counter reset; no sync, no CUDA init).

    Returns the allocated bytes at the reset, or ``None`` when CUDA is not initialized here.
    """
    if cuda is None:
        import torch
        cuda = torch.cuda
    if not cuda.is_initialized():
        return None
    cuda.reset_peak_memory_stats()
    return int(cuda.memory_allocated())


def peak_receipt(rank, transaction, at_open, cuda=None):
    """alpha-next (REPORT.md option 2): one rank's peak allocated and reserved bytes since the chunk's open.

    Receipt only, never a control-flow input; it reads max_memory_allocated/max_memory_reserved (no memory_stats call,
    so the inherited `flash-s5 memory` receipt's queries are unchanged). ``stage_timer`` says whether stock QSA_STAGE_TIMING is on: its own
    periodic peak reset then makes these values a lower bound. ``None`` when CUDA is not initialized.
    """
    if cuda is None:
        import torch
        cuda = torch.cuda
    if not cuda.is_initialized():
        return None
    return dict(rank=int(rank), txn_seq=int(transaction.txn_seq), chunk_start=int(transaction.chunk_start),
                chunk_end=int(transaction.chunk_end),
                peak_allocated=int(cuda.max_memory_allocated()),
                peak_reserved=int(cuda.max_memory_reserved()),
                allocated_at_open=-1 if at_open is None else int(at_open),
                stage_timer=int(os.environ.get('QSA_STAGE_TIMING', '0') == '1'))


def _rank(server, dispatcher):
    if server is not None:
        runtime = getattr(server, 'runtime', None)
        state = getattr(runtime, 'state', None)
        rank = getattr(state, 'worker_rank', None)
        if type(rank) is int:
            return rank
    worker = getattr(dispatcher, 'worker_rank', None)
    if type(worker) is int:
        return worker
    runner = getattr(dispatcher, 'runner', None)
    rank = getattr(runner, 'rank', None)
    if type(rank) is int:
        return rank
    raise WorkerBridgeError('native bridge cannot determine the worker rank')


def admission_to_chunk(record):
    """Exact ``AdmissionRecord`` -> ``lane_chunk.ChunkAdmission`` conversion.

    All admission fields keep their meaning. Group rows, including empty rows
    and repeated null block 0, travel unchanged; the worker discovers its own
    QSA groups. The worker-side align seed is derived later from
    ``committed_end`` and is not sent on the wire.
    """
    if record is None or not hasattr(record, 'transaction'):
        raise WorkerBridgeError('open needs an AdmissionRecord')
    return ChunkAdmission(
        transaction=record.transaction,
        prompt_end=record.prompt_end,
        input_ids=record.input_ids,
        ngram_context=record.ngram_context,
        block_tables=record.block_tables,
        new_block_ids_to_zero=record.new_block_ids_to_zero,
        cow_pairs=record.cow_pairs,
        state_epoch=record.state_epoch,
        native_end=getattr(record, 'native_end', 0),
    )


def admission_metadata(record):
    """Immutable CPU admission snapshot for the worker's lane reservation."""
    transaction = record.transaction
    row = BatchRow(transaction.request_uid, transaction.request_generation, 0,
                   transaction.chunk_start, transaction.chunk_end,
                   record.committed_end)
    return MetadataSnapshot.capture(transaction, (row,),
                                    num_computed=(record.committed_end,))


class NativeWorkerBridge:
    """One worker's typed wire-v1 front end over its real lane dispatcher."""

    def __init__(self, dispatcher, *, server=None, rank=None,
                 source_binder=None, clock=time.monotonic_ns):
        if dispatcher is None or not callable(getattr(dispatcher, 'submit', None)):
            raise WorkerBridgeError('native bridge needs the constructed LaneDispatcher')
        if not callable(getattr(dispatcher, 'open', None)):
            raise WorkerBridgeError('native bridge needs an openable lane dispatcher')
        self.dispatcher = dispatcher
        self.server = server
        self.rank = _rank(server, dispatcher) if rank is None else rank
        if type(self.rank) is not int or self.rank < 0:
            raise WorkerBridgeError('native bridge rank must be a non-negative int')
        if os.environ.get('FLASH_S5_INWAIT', '0') == '1':
            import lane_chunk
            lane_chunk.note_worker_rank(self.rank)
        self.clock = clock
        self.source_binder = source_binder
        # The worker-local pinned source binding, kept until the lane's release
        # (or a drained cancel) proves every consumer finished. It is never
        # derived from a boolean or a generic cache.
        self._source_binding = None
        self._source_close = None
        self.worker = None
        self.transaction = None
        self.lane = None
        self.state_epoch = None
        self.stage = None
        self.route_generation = 0
        self._accepted = {}
        self._host_ns = {}
        self._proofs = {}
        self._completion_seq = -1
        self._reported_seq = -1
        self._lean = None
        self._released = False
        self._runtime_released = False
        self._request_closed = False
        self._last_error = None
        # The final queried report captured while the drained lane events still
        # exist; it outlives ``_forget`` only inside ``_retained``.
        self._final_report = None
        # The last transaction this worker answered and its ReleaseAck: a
        # broadcast release retry for it is answered with the identical row
        # instead of being refused (no ownership is re-derived). A nonterminal
        # ack is retained here too, because its request row is still owned.
        self._closed = None
        # The lane request row retained by a nonterminal release, with the
        # worker's own final queried report: ``{'transaction', 'report'}``.
        # Cancellation of that chunk closes this row before acknowledging; the
        # report is what the engine's cancellation retirement re-checks.
        self._retained = None
        # One typed close tombstone, including the final queried report, for
        # identical broadcast retries after a retained row has closed.
        self._retained_cancel_ack = None
        # Set when a route switch could not be proven complete.  Every later
        # transition refuses; ownership is quarantined rather than reused.
        self._route_failure = None
        self.engine_epoch = None
        # The last replay census this worker reported to the engine.  Commands
        # carry a floor, so a floor below what was already acknowledged means
        # the engine lost a reply and must not be trusted.
        self._ack_watermark = 0
        # One exact-command ACK, retained across partial all-rank retries.
        # A same-target command with different identity is never idempotent.
        self._transition_cache = None
        # Auto-arm runs only: per-chunk allocator receipts (open, and after
        # the drained release). Never read by any control decision.
        self._memory_receipts = os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1'
        self._memory_cuda = None
        self.memory_rows = []
        self.memory_receipt_errors = 0

    # -- lifecycle ---------------------------------------------------------
    def attach(self, worker):
        """The generated GPU worker attaches itself before any typed RPC."""
        self.worker = worker
        attach = getattr(self.server, 'attach_worker', None)
        if callable(attach):
            attach(worker)
        return self

    @property
    def active(self):
        return self.transaction is not None

    def _forget(self):
        self.transaction = None
        self.lane = None
        self.state_epoch = None
        self.stage = None
        self._accepted = {}
        self._host_ns = {}
        self._proofs = {}
        self._completion_seq = -1
        self._reported_seq = -1
        self._lean = None
        self._released = False
        self._runtime_released = False
        self._request_closed = False
        self._last_error = None
        self._final_report = None
        self._source_binding = None
        self._source_close = None

    def _require_open(self, transaction, state_epoch=None):
        if self.transaction is None or self.transaction != transaction:
            raise WorkerBridgeError('no native transaction matches the request')
        if state_epoch is not None and state_epoch != self.state_epoch:
            raise WorkerBridgeError('state epoch differs from the opened chunk')

    def _native(self):
        native = getattr(self.dispatcher, 'native', None)
        if native is None:
            raise WorkerBridgeError('native bridge has no worker-native composition')
        return native

    def _worker_state(self):
        runtime = getattr(self.server, 'runtime', None)
        state = getattr(runtime, 'state', None)
        if state is None:
            state = getattr(self._native(), 'state', None)
        if state is None:
            raise WorkerBridgeError('native bridge cannot reach the worker state')
        return state

    # -- scheduler source lease -------------------------------------------
    def _prepare_resuming(self, command):
        """Build-3: a resend of a PREPARE in slices; its lease was proved and bound on the first send."""
        resuming = getattr(self.dispatcher, 'prepare_resuming', None)
        return callable(resuming) and resuming(command) is True

    def _bind_prepare_source(self, command):
        """Prove and bind the engine's actual lease before any submission.

        The descriptor is validated against the frame this worker actually
        opened, then the worker-local binder retains the real ``_qsa_host_kv``
        pages. A missing, stale, foreign or second, different lease refuses
        here, before the dispatcher can enqueue anything.
        """
        source = command.source
        if source is None:
            if self._source_binding is not None:
                raise WorkerBridgeError(
                    'prepare is missing the already-bound source lease')
            raise WorkerBridgeError('prepare carries no source lease')
        if type(source) is not SourceLeaseDescriptor:
            raise WorkerBridgeError('prepare source is not a typed lease descriptor')
        if source.transaction != self.transaction:
            raise WorkerBridgeError('source lease belongs to another transaction')
        if source.state_epoch != self.state_epoch:
            raise WorkerBridgeError('source lease state epoch is stale')
        admission = getattr(self.dispatcher, 'admission', None)
        if admission is None:
            raise WorkerBridgeError('prepare has no opened admission to prove the lease')
        try:
            source.validate_frame(admission, (self.rank,))
        except E3ContractError as exc:
            raise WorkerBridgeError('source lease was refused: ' + str(exc)) from exc
        existing = self._source_binding
        if existing is not None:
            if existing.descriptor != source:
                raise WorkerBridgeError('a different source lease is already bound')
            return existing
        binder = self.source_binder
        if binder is None:
            from native_source import bind_local_sources
            binder = bind_local_sources
        binding = binder(self.dispatcher, admission, source, rank=self.rank)
        if (binding is None
                or getattr(binding, 'descriptor', None) != source
                or getattr(binding, 'transaction', None) != self.transaction):
            raise WorkerBridgeError('source binder did not bind this exact lease')
        self._source_binding = binding
        self.dispatcher.source_binding = binding
        operation = getattr(self.dispatcher, 'operation', None)
        if operation is not None:
            operation.native_source_holds = binding.holds
            if getattr(operation, 'prefetch_enabled', False):
                operation.bind_prefetch_frame(admission)
        return binding

    def _close_source_binding(self):
        """Release worker-local source ownership only after the lane release."""
        binding = self._source_close
        if binding is None:
            binding = self.dispatcher.source_binding
        if binding is None:
            return True
        binding.close(self.dispatcher.source_registry)
        self._source_close = None
        self._source_binding = None
        if getattr(self.dispatcher, 'source_binding', None) is binding:
            self.dispatcher.source_binding = None
        return True

    # -- engine owner bootstrap -------------------------------------------
    def bootstrap(self, command):
        """Bind this worker incarnation to the engine epoch and retain storage.

        The answer is derived from the worker's own retained decode storage and
        route controller, never from a local default.  Binding is once per
        incarnation: a foreign or late epoch is refused, while a retry of the
        same epoch is idempotent and reports the current route generation.
        """
        if type(command) is not BootstrapCommand:
            raise WorkerBridgeError('bootstrap needs a typed BootstrapCommand')
        if self.rank not in command.ranks:
            raise WorkerBridgeError('bootstrap does not include this worker rank')
        state = self._worker_state()
        if state.worker_rank != self.rank:
            raise WorkerBridgeError('worker state belongs to another rank')
        owner = getattr(state, 'decode_owner', None)
        handle = getattr(getattr(state, 'decode', None), 'handle', None)
        if owner is None or handle is None:
            raise WorkerBridgeError('worker has no retained decode storage')
        if state.decode.state != 'retained':
            raise WorkerBridgeError('decode storage is not persistently retained')
        if owner.lane_id != handle.lane_id:
            raise WorkerBridgeError('retained decode owner differs from the lane handle')
        state.bind_engine_owner(command.engine_epoch, incarnation=state.incarnation)
        router = getattr(self._native(), 'router', None)
        if router is not None:
            router.bind_engine_owner(command.engine_epoch, incarnation=state.incarnation)
        controller = self._native().controller
        self.engine_epoch = command.engine_epoch
        watermark = getattr(router, 'replay_watermark', 0)
        self._ack_watermark = max(self._ack_watermark, watermark)
        return BootstrapAck(command.engine_epoch, self.rank, state.incarnation,
                            owner.lane_id, controller.route_generation,
                            watermark, True)

    # -- open --------------------------------------------------------------
    def open(self, request):
        if type(request) is not OpenRequest:
            raise WorkerBridgeError('open needs a typed OpenRequest')
        if self.transaction is not None:
            raise WorkerBridgeError('a native transaction is already open')
        self._closed = None
        if self.rank not in request.ranks:
            raise WorkerBridgeError('open request does not include this worker')
        if self.server is None:
            raise WorkerBridgeError('native bridge needs the worker handoff server')
        record = request.admission
        if self._retained is not None and not self._retained['transaction'].same_request(
                record.transaction):
            # The retained row belongs to another request; opening would reuse
            # or overwrite the lane row of a request that never closed.
            raise WorkerBridgeError('a retained request row belongs to another request')
        self._retained_cancel_ack = None
        if self._memory_receipts:
            self._memory_receipt('open', record.transaction)
            self._peak_open()
        chunk_admission = admission_to_chunk(record)
        snapshot = admission_metadata(record)
        ack = self.server.open(OpenPrefill(record.transaction, chunk_admission.num_tokens,
                                           metadata=snapshot))
        lane = getattr(ack, 'lane', None)
        if lane is None:
            handle = getattr(getattr(self.server, 'runtime', None), 'state', None)
            handle = getattr(getattr(handle, 'prefill', None), 'handle', None)
            lane = getattr(handle, 'lane_id', None)
        if lane is None or lane.worker_rank != self.rank or lane.role.value != 'prefill':
            raise WorkerBridgeError('worker open did not bind this prefill lane')
        operation = getattr(self.server, 'operation', None)
        if operation is None:
            raise WorkerBridgeError('worker open produced no native operation')
        self.dispatcher.open(record.transaction, chunk_admission, operation)
        if request.qsa_fused:
            self.dispatcher.select_fused()
        if request.layer_group != 1:
            self.dispatcher.select_layer_group(request.layer_group)
        stage = self.dispatcher.stage
        if stage not in (0, 1):
            raise WorkerBridgeError('dispatcher stage is outside the two-stage split')
        self.transaction = record.transaction
        self.lane = lane
        self.state_epoch = record.state_epoch
        self.stage = stage
        self._lean = _LeanSent() if request.replies == 'lean' else None
        # The open moved the retained row into the active transaction; the row
        # is now released/closed through the normal paths below.
        self._retained = None
        pp = getattr(getattr(self.dispatcher, 'native', None), 'comms', None)
        pp_stages = 2 if getattr(pp, 'pp', None) is not None else 1
        return OpenAck(record.transaction, lane, stage, record.state_epoch,
                       pp_stages, True)

    # -- submission --------------------------------------------------------
    def submit(self, command):
        if type(command) is not SegmentCommand:
            raise WorkerBridgeError('submit needs a typed SegmentCommand')
        self._require_open(command.transaction)
        report, cursor = self._reply_report()
        if self.rank not in command.ranks:
            return SegmentAcceptance(
                command.transaction, self.rank, command.kind, command.stage,
                command.op_seq, skipped=True, cursor=cursor, report=self._sent(report))
        if command.kind is SegmentKind.PREPARE and not self._prepare_resuming(command):
            self._bind_prepare_source(command)
        if command.profile is not None:
            if command.profile.fused:
                self._fused_prepare_command(command.profile)
            else:
                self._profile_boundary_command(command.profile.query)
        try:
            acknowledgement = self.dispatcher.submit(command)
        except E3Pending:
            return SegmentAcceptance(
                command.transaction, self.rank, command.kind, command.stage,
                command.op_seq, pending=True, cursor=cursor, report=self._sent(report))
        except BaseException as exc:
            if getattr(self.dispatcher, 'quarantined', None) is not None:
                self._last_error = type(exc).__name__
            raise
        self._accepted[command.op_seq] = command.kind
        elapsed = acknowledgement.get('host_submit_ns')
        self._host_ns[command.op_seq] = (int(elapsed)
                                         if type(elapsed) in (int, float) and elapsed >= 0
                                         else 0)
        result = acknowledgement.get('result') or {}
        report, cursor = self._reply_report()
        return SegmentAcceptance(
            command.transaction, self.rank, command.kind, command.stage,
            command.op_seq, accepted=True, cursor=cursor,
            host_submit_ns=self._host_ns[command.op_seq],
            paused=bool(result.get('paused')), report=self._sent(report),
            profile_boundary=bool(result.get('profile_boundary')),
            tail_boundary=bool(result.get('tail_boundary')),
            dense_profiles=result.get('dense_profiles'),
            collectives=acknowledgement.get('collectives'))

    # -- peer binding / receipts ------------------------------------------
    def bind_pp(self, binding):
        if type(binding) is not PeerBinding:
            raise WorkerBridgeError('bind needs a typed PeerBinding')
        self._require_open(binding.transaction)
        if binding.sender_rank != self.rank:
            raise WorkerBridgeError('peer binding targets another sender rank')
        if self.stage != 0:
            raise WorkerBridgeError('only the sending stage binds a PP consumer')
        channel = getattr(getattr(self.dispatcher, 'native', None), 'comms', None)
        channel = getattr(channel, 'pp', None)
        if channel is None or not getattr(channel, 'is_sender', False):
            raise WorkerBridgeError('worker has no sending PP channel to bind')
        existing = getattr(channel, 'consumer_binding', None)
        wanted = (binding.receiver_lane, binding.route_generation)
        if existing != wanted:
            channel.bind_consumer(binding.receiver_lane, binding.route_generation)
        self.route_generation = binding.route_generation
        return PeerBindingAck(binding.transaction, self.rank, binding.receiver_lane,
                              binding.route_generation, True)

    # The generated method is ``flash_s5_bind``; the handler name is the wire
    # name. The public adapter method keeps the explicit ``_pp`` suffix.
    bind = bind_pp

    def receipt(self, delivery):
        if type(delivery) is not PPReceiptDelivery:
            raise WorkerBridgeError('receipt needs a typed PPReceiptDelivery')
        self._require_open(delivery.transaction)
        if delivery.sender_rank != self.rank:
            raise WorkerBridgeError('receipt delivery targets another sender rank')
        for receipt in delivery.receipts:
            self.dispatcher.accept_pp_consumer_proof(receipt)
        return PPReceiptAck(delivery.transaction, self.rank, delivery.seqs, True)

    # -- completion --------------------------------------------------------
    def poll(self, request):
        if type(request) is QsaProfileQuery:
            self._require_open(request.transaction, request.state_epoch)
            if self.rank not in request.ranks:
                if self.stage == request.stage:
                    raise WorkerBridgeError('QSA profile omitted a local-stage peer')
                return QsaProfileObservation(request, self.rank, 'skipped')
            self._profile_boundary_command(request)
            if max(self.dispatcher.accepted) != request.boundary_op_seq:
                raise WorkerBridgeError('QSA profile query follows another submission')
            return self.dispatcher.driver.schedule_profile(request, self.rank)
        if type(request) is not PollRequest:
            raise WorkerBridgeError('poll needs a typed PollRequest')
        self._require_open(request.transaction, request.state_epoch)
        if self._lean is not None:
            # A poll reply is the report itself: always a typed lean report.
            return self._sent(self._lean_report(token=False)[0])
        return self._build_report()

    def _profile_boundary_command(self, query):
        self._require_open(query.transaction, query.state_epoch)
        if query.stage != self.stage or self.rank not in query.ranks:
            raise WorkerBridgeError('QSA profile belongs to another stage')
        stored = self.dispatcher.accepted.get(query.boundary_op_seq)
        if stored is None:
            raise WorkerBridgeError('QSA profile has no accepted submission boundary')
        command, ack = stored
        if (command.kind != 'layer' or command.layer_start != query.layer
                or not (ack.get('result') or {}).get('profile_boundary')):
            raise WorkerBridgeError('QSA profile does not name its accepted layer boundary')

    def _fused_prepare_command(self, grant):
        """A fused grant names this worker's own accepted PREPARE and its reported plan."""
        query = grant.query
        self._require_open(query.transaction, query.state_epoch)
        if query.stage != self.stage or self.rank not in query.ranks:
            raise WorkerBridgeError('fused grant belongs to another stage')
        stored = self.dispatcher.accepted.get(query.boundary_op_seq)
        if stored is None or stored[0].kind != 'prepare':
            raise WorkerBridgeError('fused grant has no accepted PREPARE')
        reported = (stored[1].get('result') or {}).get('dense_profiles')
        if reported is None or dict(reported).get(query.layer) != grant.shape:
            raise WorkerBridgeError('fused grant differs from the reported PREPARE plan')

    def _proof(self, op_seq, kind):
        key = (op_seq, kind)
        proof = self._proofs.get(key)
        if proof is None:
            role = _SEGMENT_ROLE[self._accepted[op_seq]]
            ticket = OperationTicket(self.transaction, self.rank, self.lane,
                                     op_seq, role)
            self._completion_seq += 1
            proof = CompletionProof(ticket, kind, self._completion_seq)
            self._proofs[key] = proof
        return proof

    def _cursor(self):
        if self.dispatcher.chunk is None or self.dispatcher.lane is None:
            return CompletionCursor('idle', 0, 0)
        state = getattr(self.dispatcher.lane, 'state', None)
        if state == 'finished':
            written = self.dispatcher.state_event is not None and bool(
                self.dispatcher.state_event.query())
            phase = 'commit' if written else 'state_finalize'
        elif state == 'running':
            phase = 'layer'
        else:
            phase = 'prepare'
        layer = getattr(self.dispatcher.lane, 'cursor', 0) or 0
        if type(layer) is not int or layer < 0:
            layer = 0
        bucket = 0
        inner = self.dispatcher.cursor()
        if inner is not None and type(inner[2]) is tuple and len(inner[2]) >= 3:
            value = inner[2][2]
            if type(value) is int and value >= 0:
                bucket = value
        return CompletionCursor(phase, layer, bucket)

    def _build_report(self):
        query = self.dispatcher.poll()
        completed = tuple(sorted(query['completed_op_seqs']))
        pending = tuple(sorted(query['pending_op_seqs']))
        proofs = []
        for op_seq in completed:
            kind = self._accepted.get(op_seq)
            if kind is None:
                # A completion from an earlier transaction is never exported
                # under this report's identity.
                continue
            proofs.append(self._proof(op_seq, CompletionKind.GPU_EVENT))
            if kind is SegmentKind.STATE_FINALIZE and query['state_written']:
                proofs.append(self._proof(op_seq, CompletionKind.STATE_WRITE))
            if kind is SegmentKind.PP_RECEIVE and self.dispatcher.pp_consumer_complete():
                proofs.append(self._proof(op_seq, CompletionKind.PP_CONSUMER))
        hosts = tuple(sorted(self._host_ns.items()))
        self._reported_seq = max(self._reported_seq, self._completion_seq)
        watermark = max(0, self._reported_seq)
        return WorkerCompletionReport(
            self.transaction, self.rank, completed, pending, watermark,
            tuple(proofs), self._cursor(), self.state_epoch, host_submit_ns=hosts,
            failed_code=None if getattr(self.dispatcher, 'quarantined', None) is None
            else 'quarantined')

    def _reply_report(self):
        """(report, cursor) a reply carries now; a lean report is sent only through _sent."""
        if self._lean is None:
            report = self._build_report()
            return report, report.cursor
        return self._lean_report()

    def _lean_report(self, token=True):
        """Held-g L4: the delta since this rank's last sent lean report, not yet sent.

        Proofs come in a full report's order (completed op_seq, then GPU_EVENT,
        STATE_WRITE, PP_CONSUMER) over the new and the owed completions, so
        ``_proof`` assigns the same completion_seq a full build would. With
        ``token`` and nothing changed since the last sent report, the reply
        carries no report (None).
        """
        sent = self._lean
        query = self.dispatcher.poll()
        done = sent.completed
        new = tuple(sorted(op for op in query['completed_op_seqs'] if op not in done))
        pending = tuple(sorted(query['pending_op_seqs']))
        keys = sent.proof_keys
        proofs = []
        for op_seq in sorted(sent.due.union(new)) if sent.due else new:
            kind = self._accepted.get(op_seq)
            if kind is None:
                # A completion from an earlier transaction is never exported
                # under this report's identity.
                continue
            wanted = [CompletionKind.GPU_EVENT]
            if kind is SegmentKind.STATE_FINALIZE and query['state_written']:
                wanted.append(CompletionKind.STATE_WRITE)
            if kind is SegmentKind.PP_RECEIVE and self.dispatcher.pp_consumer_complete():
                wanted.append(CompletionKind.PP_CONSUMER)
            for proof_kind in wanted:
                proof = self._proof(op_seq, proof_kind)
                if (op_seq, proof_kind) not in keys:
                    proofs.append(proof)
        hosts = tuple(entry for entry in sorted(self._host_ns.items())
                      if entry[0] not in sent.hosts)
        watermark = max(0, self._reported_seq, self._completion_seq)
        cursor = self._cursor()
        failed = (None if getattr(self.dispatcher, 'quarantined', None) is None
                  else 'quarantined')
        proofs = tuple(proofs)
        if token and not new and not proofs and not hosts and sent.last == (
                pending, watermark, cursor, failed):
            return None, cursor
        bases = (len(done), len(keys), len(sent.hosts))
        return LeanCompletionReport(
            self.transaction, self.rank, *bases, new, pending,
            watermark, proofs, cursor, self.state_epoch,
            lean_digest(sent.digest, bases, new, proofs, hosts),
            host_submit_ns=hosts, failed_code=failed), cursor

    def _sent(self, report):
        """Mark a lean report as sent (the reply carrying it is being returned)."""
        if type(report) is not LeanCompletionReport:
            return report
        sent = self._lean
        if report.bases() != (len(sent.completed), len(sent.proof_keys), len(sent.hosts)):
            raise WorkerBridgeError('lean report was built on another sent state')
        sent.completed.update(report.completed_op_seqs)
        for proof in report.proofs:
            sent.proof_keys.add((proof.ticket.op_seq, proof.kind))
        for op_seq in report.completed_op_seqs:
            kind = self._accepted.get(op_seq)
            if kind is SegmentKind.STATE_FINALIZE or kind is SegmentKind.PP_RECEIVE:
                sent.due.add(op_seq)
        for op_seq in tuple(sent.due):
            owed = (CompletionKind.STATE_WRITE
                    if self._accepted.get(op_seq) is SegmentKind.STATE_FINALIZE
                    else CompletionKind.PP_CONSUMER)
            if (op_seq, owed) in sent.proof_keys:
                sent.due.discard(op_seq)
        sent.hosts.update(op_seq for op_seq, _ns in report.host_submit_ns)
        sent.digest = report.digest
        sent.last = (report.pending_op_seqs, report.completed_sequence, report.cursor,
                     report.failed_code)
        self._reported_seq = max(self._reported_seq, report.completed_sequence)
        return report

    # -- closure -----------------------------------------------------------
    def _drain_prefill_streams(self):
        """Retire owned side-stream work before releasing the lane's storage.

        Older CPU/diagnostic ports have no native side-stream owner. Real
        native construction supplies it through the dispatcher; this hook
        does not establish that construction or grant native readiness.
        """
        native_getter = getattr(self, '_native', None)
        native = (native_getter() if callable(native_getter)
                  else getattr(self.dispatcher, 'native', None))
        streams = getattr(native, 'prefill_streams', None)
        drain = getattr(streams, 'drain', None)
        if not callable(drain):
            return 0
        return drain()

    def _memory_receipt(self, point, transaction):
        """Bounded receipt: one log line; a failure is counted, never raised."""
        try:
            row = memory_receipt(point, self.rank, transaction, cuda=self._memory_cuda)
            if row is None:
                return
            self.memory_rows = (self.memory_rows + [row])[-16:]
            import logging
            # Alpha-next PLAN 2: one atomic write through the process receipt writer.
            from native_measurement import emit_receipt_kv
            emit_receipt_kv(
                'flash-s5 memory point=%s rank=%d txn_seq=%d chunk=%d-%d allocated=%d '
                'reserved=%d inactive_split=%d alloc_retries=%d ooms=%d free=%d total=%d '
                'sync_all_streams=%d device_allocs=%d device_frees=%d'
                % (row['point'], row['rank'], row['txn_seq'], row['chunk_start'],
                   row['chunk_end'], row['allocated'], row['reserved'], row['inactive_split'],
                   row['alloc_retries'], row['ooms'], row['free'], row['total'],
                   row['sync_all_streams'], row['device_allocs'], row['device_frees']),
                logging.getLogger(__name__))
        except Exception:
            self.memory_receipt_errors += 1

    def _cow_receipt(self, transaction):
        """inwait build-4: this chunk's zero and CoW times, one line; the lane is drained here. A failure is counted, never raised."""
        if os.environ.get('FLASH_S5_INWAIT', '0') != '1':
            return
        import lane_chunk
        try:
            row = lane_chunk.take_cow_row()
            if row is None:
                return
            row = lane_chunk.cow_row_pages(lane_chunk.cow_row_times(row), transaction.chunk_start)
            import logging
            from native_measurement import emit_receipt_kv
            written = emit_receipt_kv('flash-s5 cow rank=%d txn_seq=%d chunk=%d-%d %s' % (
                self.rank, transaction.txn_seq, transaction.chunk_start, transaction.chunk_end,
                ' '.join('%s=%s' % (key, row[key]) for key in sorted(row))), logging.getLogger(__name__))
            if written is False:
                # build-5b: the writer refused it; build-4c lost every line this way, silently.
                lane_chunk.COW_FAST['row_errors'] += 1
        except Exception:
            lane_chunk.COW_FAST['row_errors'] += 1

    def _peak_open(self):
        """alpha-next: reset the allocator peak at the chunk's open; a failure is counted (peak_receipt_errors), never raised."""
        try:
            self._peak_at_open = peak_reset(cuda=self._memory_cuda)
        except Exception:
            self._peak_at_open = None
            self.peak_receipt_errors = getattr(self, 'peak_receipt_errors', 0) + 1

    def _peak_receipt(self, transaction):
        """alpha-next: the chunk's peak-allocated receipt line; a failure is counted (peak_receipt_errors), never raised."""
        try:
            row = peak_receipt(self.rank, transaction, getattr(self, '_peak_at_open', None),
                               cuda=self._memory_cuda)
            if row is None:
                return
            import logging
            # Alpha-next PLAN 2: one atomic write through the process receipt writer.
            from native_measurement import emit_receipt_kv
            emit_receipt_kv(
                PEAK_LINE + 'rank=%d txn_seq=%d chunk=%d-%d peak_allocated=%d peak_reserved=%d '
                'allocated_at_open=%d stage_timer=%d' % (
                    row['rank'], row['txn_seq'], row['chunk_start'], row['chunk_end'], row['peak_allocated'],
                    row['peak_reserved'], row['allocated_at_open'], row['stage_timer']),
                logging.getLogger(__name__))
        except Exception:
            self.peak_receipt_errors = getattr(self, 'peak_receipt_errors', 0) + 1

    def _route_fail(self, reason):
        self._route_failure = reason
        router = getattr(self._native(), 'router', None)
        if router is not None and not getattr(router, 'unclean', False):
            try:
                router.quarantine(reason)
            except E3ContractError:
                pass
        return WorkerBridgeError(reason)

    def _release_lane(self, transaction, terminal):
        """Idempotent drain/release; close the request row only when terminal.

        A nonterminal release drops the chunk's storage and leaves the request
        row so the next chunk's ``LaneChunkFactory.admit`` reaches
        ``LaneRequest.update()``. A terminal release (or any cancellation)
        closes the row before the acknowledgment, so stock can never observe a
        row this lane still owns.
        """
        if not self._released:
            if not self.dispatcher.drain():
                return False
            self._drain_prefill_streams()
            # Query the drained lane now: release() clears its events, and a
            # nonterminal request row must carry its own final report for the
            # engine's later cancellation retirement.
            self._final_report = self._build_report()
            # ``release`` resets the dispatcher's per-transaction fields, so
            # the binding is captured first and closed only after the lane
            # itself proved every consumer drained.
            binding = getattr(self.dispatcher, 'source_binding', None)
            self.dispatcher.release()
            self._source_close = binding
            self._close_source_binding()
            self._released = True
            if self._memory_receipts:
                self._memory_receipt('released', transaction)
                self._peak_receipt(transaction)
            self._cow_receipt(transaction)
        if not self._runtime_released:
            runtime = getattr(self.server, 'runtime', None)
            release = getattr(runtime, 'release', None)
            if callable(release):
                release(transaction)
            self._runtime_released = True
        if terminal and not self._request_closed:
            self.dispatcher.close_request()
            self._request_closed = True
        clear = getattr(self.server, 'clear_native', None)
        if callable(clear):
            clear(transaction)
        return True

    def release(self, request):
        if type(request) is not ReleaseRequest:
            raise WorkerBridgeError('release needs a typed ReleaseRequest')
        if self.transaction is None and self._closed is not None:
            prior = self._closed
            if (prior.transaction == request.transaction
                    and prior.state_epoch == request.state_epoch
                    and prior.request_closed == request.terminal
                    and dict(request.completed_sequences).get(self.rank)
                    == prior.completed_sequence):
                return prior
        self._require_open(request.transaction, request.state_epoch)
        chunk = self.dispatcher.chunk
        if chunk is not None:
            # coverage-soak-fix: the chunk ending at the admitted native end (the prompt's last row, or a geometry tail's
            # block boundary) is the terminal one
            admitted_terminal = chunk.admission.chunk_end == chunk.admission.native_end
            if request.terminal is not admitted_terminal:
                raise WorkerBridgeError(
                    'release terminal flag differs from the admitted chunk')
        expected = dict(request.completed_sequences)
        if self.rank not in expected:
            raise WorkerBridgeError('release does not cover this rank')
        if expected[self.rank] != max(0, self._reported_seq):
            raise WorkerBridgeError('release watermark differs from the final report')
        if self._lean is not None:
            # Held-g L4: the engine applied exactly the stream this rank sent,
            # and the drained lane has nothing left to report.
            streams = {entry[0]: tuple(entry[1:]) for entry in request.lean_streams}
            if streams.get(self.rank) != self._lean.end():
                raise WorkerBridgeError('lean release differs from the sent report stream')
            rest, _cursor = self._lean_report(token=False)
            if not rest.empty():
                raise WorkerBridgeError('lean release leaves completions unreported')
        elif request.lean_streams:
            raise WorkerBridgeError('lean release streams on a full-reply transaction')
        try:
            closed = self._release_lane(request.transaction, request.terminal)
        except E3Pending:
            closed = False
        except BaseException as exc:
            self._last_error = type(exc).__name__
            raise
        ack = ReleaseAck(request.transaction, self.rank, self.state_epoch,
                         expected[self.rank], released=closed,
                         request_closed=closed and request.terminal)
        if closed:
            if not request.terminal:
                if self._final_report is None:
                    raise WorkerBridgeError(
                        'nonterminal release lost its final queried report')
                self._retained = dict(transaction=request.transaction,
                                      report=self._final_report)
            else:
                self._retained = None
            self._forget()
            self._closed = ack
        return ack

    def _still_owned(self, transaction):
        """Server/dispatcher state left by a partial open of ``transaction``."""
        if getattr(self.dispatcher, 'transaction', None) == transaction:
            return True
        handoff = getattr(self.server, 'handoff', None)
        if handoff is None:
            return False
        return handoff == transaction or getattr(handoff, 'transaction', None) == transaction

    def cancel(self, request):
        if type(request) is not CancelRequest:
            raise WorkerBridgeError('cancel needs a typed CancelRequest')
        if self.transaction is None:
            if self._still_owned(request.transaction):
                # A partially opened worker (server or dispatcher opened, the
                # bridge never bound) owns storage it cannot certify closed.
                return CancelAck(request.transaction, self.rank, request.state_epoch,
                                 'quarantined', (), (), 0)
            prior_cancel = self._retained_cancel_ack
            if (self._retained is None and prior_cancel is not None
                    and prior_cancel.transaction == request.transaction
                    and prior_cancel.state_epoch == request.state_epoch):
                return prior_cancel
            prior = self._closed
            if (prior is not None and prior.closed
                    and prior.transaction == request.transaction
                    and prior.state_epoch == request.state_epoch):
                # This transaction already certified terminal closure (a stop
                # raced its release); the broadcast confirms, never re-derives.
                return CancelAck(request.transaction, self.rank, request.state_epoch,
                                 'released', (), (), prior.completed_sequence,
                                 released=True, request_closed=True)
            row = self._retained
            if row is not None:
                report = row['report']
                if (row['transaction'] == request.transaction
                        and report is not None
                        and report.state_epoch == request.state_epoch):
                    # A nonterminal release left this request row open. The
                    # request is now being cancelled, so the row closes before
                    # the terminal acknowledgment.
                    try:
                        self.dispatcher.close_request()
                    except BaseException as exc:
                        self._last_error = type(exc).__name__
                        return CancelAck(request.transaction, self.rank,
                                         request.state_epoch, 'quarantined',
                                         report.completed_op_seqs,
                                         report.pending_op_seqs,
                                         report.completed_sequence, report=report)
                    self._retained = None
                    self._closed = None
                    ack = CancelAck(request.transaction, self.rank, request.state_epoch,
                                     'released', report.completed_op_seqs,
                                     report.pending_op_seqs,
                                     report.completed_sequence,
                                     released=True, request_closed=True,
                                     report=report)
                    self._retained_cancel_ack = ack
                    return ack
                # The retained row belongs to another chunk/request: do not
                # guess which row this cancellation owns.
                return CancelAck(request.transaction, self.rank, request.state_epoch,
                                 'quarantined', (), (),
                                 report.completed_sequence if report is not None else 0)
            watermark = max(0, self._reported_seq)
            return CancelAck(request.transaction, self.rank, request.state_epoch,
                             'never_open', (), (), watermark,
                             released=True, request_closed=True)
        self._require_open(request.transaction, request.state_epoch)
        transaction = self.transaction
        self.dispatcher.cancel(request.reason)
        report = self._build_report()
        completed = report.completed_op_seqs
        pending = report.pending_op_seqs
        watermark = report.completed_sequence
        if getattr(self.dispatcher, 'quarantined', None) is not None:
            return CancelAck(transaction, self.rank, self.state_epoch,
                             'quarantined', completed, pending, watermark,
                             report=report)
        disposition = 'accepted' if self._accepted else 'opened'
        try:
            # Cancellation is always terminal: the request row closes before
            # the acknowledgment, and the retained report is dropped with it.
            closed = self._release_lane(transaction, True)
        except E3Pending:
            closed = False
        except BaseException as exc:
            self._last_error = type(exc).__name__
            return CancelAck(transaction, self.rank, self.state_epoch,
                             'quarantined', completed, pending, watermark,
                             report=report)
        if closed:
            # drain() may first establish the cancelled receiver's consumer
            # completion. Preserve that final queried PP proof before reset;
            # the pre-drain report cannot release the remote sender's slot.
            report = self._final_report or report
            ack = CancelAck(transaction, self.rank, request.state_epoch,
                            'released', report.completed_op_seqs,
                            report.pending_op_seqs, report.completed_sequence,
                            released=True, request_closed=True, report=report)
            self._forget()
            self._retained_cancel_ack = ack
            return ack
        return CancelAck(transaction, self.rank, request.state_epoch,
                         disposition, completed, pending, watermark, report=report)

    def transition(self, command):
        """Drained full <-> split switch, acknowledged only after it applied.

        Every field is proven against this worker's own retained storage, route
        controller and queried replay census.  A refused command mutates
        nothing; an unexpected failure after mutation starts quarantines this
        worker's route ownership instead of claiming a partial success.
        """
        if type(command) is not TransitionCommand:
            raise WorkerBridgeError('transition needs a typed TransitionCommand')
        if self._route_failure is not None:
            raise WorkerBridgeError('route ownership is quarantined: ' + self._route_failure)
        state = self._worker_state()
        native = self._native()
        controller = getattr(native, 'controller', None)
        router = getattr(native, 'router', None)
        owner = getattr(state, 'decode_owner', None)
        handle = getattr(getattr(state, 'decode', None), 'handle', None)
        if controller is None or router is None:
            raise WorkerBridgeError('route transition needs the constructed controller and router')
        if not state.engine_bound or self.engine_epoch is None:
            raise WorkerBridgeError('route transition before the engine owner bootstrap')
        if command.engine_epoch != state.engine_epoch:
            raise WorkerBridgeError('transition command names another engine epoch')
        if self.transaction is not None or self._retained is not None:
            raise WorkerBridgeError('route transition while a request row is still owned')
        if self.rank >= len(command.storage_epochs):
            raise WorkerBridgeError('transition command does not cover this rank')
        if owner is None or handle is None or owner.lane_id != handle.lane_id:
            raise WorkerBridgeError('persistent decode storage owner changed')
        if state.decode.state != 'retained':
            raise WorkerBridgeError('decode storage is not persistently retained')
        if command.storage_epochs[self.rank] != handle.storage_epoch:
            raise WorkerBridgeError('transition command names a stale storage epoch')
        cached = self._transition_cache
        if cached is not None and command == cached[0]:
            ack = cached[1]
            route = 'split' if command.target_bank == ROUTE_BANK_SPLIT else 'full'
            mode = 'decode-split' if route == 'split' else 'ordinary'
            router.poll_native_replays()
            if (controller.route != route
                    or controller.route_generation != command.route_generation
                    or router.current_mode != mode
                    or router.route_generation != command.route_generation
                    or router.pending_replays
                    or router.replay_watermark != ack.decode_replay_watermark):
                raise self._route_fail('applied transition changed before its exact retry')
            return ack
        commanded = command.decode_replay_watermarks[self.rank]
        if commanded < self._ack_watermark:
            # The engine lost an earlier acknowledgment and is driving this
            # rank from a census this worker already passed.
            raise WorkerBridgeError('transition replay floor predates the acknowledged census')
        if commanded > router.replay_watermark:
            raise WorkerBridgeError('transition replay floor is ahead of the queried census')
        if command.expected_route_generation != controller.route_generation:
            raise WorkerBridgeError('transition command expects another route generation')
        if command.route_generation != controller.route_generation + 1:
            raise WorkerBridgeError('transition command must advance the route generation by one')
        if command.target_bank == ROUTE_BANK_SPLIT:
            route, router_target = 'split', 'decode-split'
            target_context = owner.context_id
        elif command.target_bank == ROUTE_BANK_FULL:
            route, router_target = 'full', 'ordinary'
            target_context = router.full_context_id
        else:
            raise WorkerBridgeError('unknown transition target bank')
        if route == controller.route:
            raise WorkerBridgeError('transition would not change the worker route')
        authorization = TransitionAuthorization(
            command.engine_epoch, target_context, handle.storage_epoch,
            command.route_generation, 'drained native route switch')
        # Retire already-queried native replays.  If any are still in flight,
        # the consumer drain is the real GPU barrier (the replay event is on
        # the green decode stream); query again afterwards and only then make
        # the non-mutating route proof.  A still-pending event stays pending
        # and refuses the switch instead of being acknowledged or cleared.
        router.poll_native_replays()
        try:
            router.validate_transition(router_target, authorization)
        except E3Pending:
            controller.drain()
            router.poll_native_replays()
            try:
                router.validate_transition(router_target, authorization)
            except E3Pending:
                return TransitionPending(command, self.rank,
                                         controller.route_generation)
        try:
            # Prefill side streams are part of the drained route: no queued
            # stock SharedExperts/PLE work may cross the bank switch.
            self._drain_prefill_streams()
        except BaseException as exc:
            raise self._route_fail(
                'prefill side streams did not drain: ' + type(exc).__name__) from exc
        try:
            before = getattr(native, 'before_route_change', None)
            if callable(before):
                before(route)
            record = controller.enter_split() if route == 'split' else controller.leave_split()
        except BaseException as exc:
            raise self._route_fail(
                'route switch failed; stream/bank state is ambiguous: '
                + type(exc).__name__) from exc
        if controller.route != route or controller.route_generation != command.route_generation:
            raise self._route_fail('route controller did not apply the requested switch')
        try:
            router.transition(router_target, authorization)
        except BaseException as exc:
            raise self._route_fail(
                'graph router did not apply the switched route: '
                + type(exc).__name__) from exc
        if router.current_mode != router_target:
            raise self._route_fail('graph router mode differs from the switched route')
        if router.replay_watermark < commanded:
            raise self._route_fail('replay census went backwards during the route switch')
        self._ack_watermark = max(self._ack_watermark, router.replay_watermark)
        self.route_generation = controller.route_generation
        ack = TransitionAck(
            command.engine_epoch, command.expected_route_generation,
            command.route_generation, command.target_bank, command.storage_epochs,
            controller.route_generation, command.target_bank, command.storage_epochs,
            worker_rank=self.rank,
            decode_replay_watermark=router.replay_watermark,
            drained_streams=record.drained_streams)
        self._transition_cache = (command, ack)
        return ack


__all__ = ['NativeWorkerBridge', 'WorkerBridgeError', 'admission_metadata',
           'admission_to_chunk']
