"""Worker-side lane dispatcher: typed segments -> bounded host operations (D4/D2/D3 seam).

Replaces D2's refusing ``_execute_prefill`` seam. One transaction at a time per
worker. The engine (Codex's driver) sends one segment command per tick; every
command is a bounded host operation under the lane scopes (prefill stream,
inference mode, lane forward context, PyNccl-only TP communicator, private PLE
scratch) and returns after enqueueing. Nothing here waits on the device.

Segment kinds (wire v1 names; the descriptor type is Codex's, consumed through
``SegmentCommand.from_descriptor``):

``prepare``         admit the chunk on this worker: register/update the request
                    row, build private inputs/metadata, install the QSA
                    operation on the stage's QSA layers (D2's seam);
``pp_receive``      stage 1 only: pre-post the HC receive for this chunk;
``prefix``          embedding (stage 0) or the received HC tensor (stage 1);
``layer``           one decoder layer, or one bounded phase of a QSA layer
                    (``QsaLayerDriver``); a paused layer keeps the cursor and
                    is resubmitted by the engine;
``suffix``          stage 0 materializes the HC tensor for PP; stage 1 no-op;
``pp_send``         stage 0 only: enqueue header+payload on the lane PP channel;
``state_finalize``  align post-process + computed count, then one STATE_WRITE
                    event on the prefill stream after joining the copy stream;
                    closes the QSA operation's forward (D2's ``end_forward``).

Completion is host-queried from the recorded events (``poll``); the report
distinguishes segment events, the state-write event and PP receive readiness.
Cancel stops new submissions and keeps every owner until ``drain`` is true.

Wire ordering: every accepted submission carries a new, strictly increasing
``op_seq``.  Resubmitting an accepted ``op_seq`` with the identical command is
idempotent (the stored acknowledgement is returned and nothing executes); any
other reuse is refused.  An :class:`E3Pending` handler result consumes no
``op_seq`` and records no event, so the engine may retry it.  The PP transfer
sequence is a separate ``seq`` field, required for ``pp_send``/``pp_receive``
and refused for every other kind.

Release is proof-gated: the worker's QSA operation must report drained, a
receiver must have recorded its consumer after state finalization and that
event must be host-queried complete, and a sender must have local send
completion plus the engine's identity-matched ``PP_CONSUMER`` proof.  Cancel
never bypasses these checks.  A handler or event-recording failure after a
partial enqueue quarantines the dispatcher: further submits are refused,
``drain`` is always false and ``release`` refuses.
"""
import time
from copy import deepcopy
from contextlib import ExitStack, nullcontext

from e3_types import E3ContractError, E3Pending, TransactionId
from lane_comm import CollectiveLedger, LaneCommError

KINDS = ('prepare', 'pp_receive', 'prefix', 'layer', 'suffix', 'pp_send', 'state_finalize')
PP_KINDS = ('pp_receive', 'pp_send')
_UNSET = object()


class LaneDispatchError(E3ContractError):
    pass


class SegmentCommand:
    """Local view of one wire-v1 SegmentDescriptor.

    ``seq`` is the PP transfer sequence; it is mandatory for the two PP kinds
    and must be absent for all others (there is no implicit zero).

    ``source`` is the wire-v1 :class:`SourceLeaseDescriptor` the engine attached
    to a PREPARE command. It is carried unchanged so the worker bridge can
    validate it against the frame it actually opened before any submission.
    """

    __slots__ = ('transaction', 'stage', 'layer_start', 'layer_end', 'op_seq',
                 'kind', 'seq', 'source', 'profile', 'layer_group')

    def __init__(self, transaction, stage, kind, op_seq, *, layer_start=None, layer_end=None,
                 seq=_UNSET, source=None, profile=None, layer_group=1):
        if type(transaction) is not TransactionId:
            raise LaneDispatchError('canonical transaction required')
        if type(kind) is not str or kind not in KINDS:
            raise LaneDispatchError(f'unknown segment kind {kind!r}')
        for name, value in (('stage', stage), ('op_seq', op_seq)):
            if type(value) is not int or value < 0:
                raise LaneDispatchError(f'segment {name} must be a non-negative int')
        if stage not in (0, 1):
            raise LaneDispatchError('only two PP stages supported')
        if kind != 'layer' and (layer_start is not None or layer_end is not None):
            raise LaneDispatchError('non-layer command carries layer bounds')
        if kind in PP_KINDS:
            if seq is _UNSET:
                raise LaneDispatchError(f'{kind} needs an explicit transfer seq')
            if type(seq) is not int or seq < 0:
                raise LaneDispatchError('segment seq must be a non-negative int')
        else:
            if seq is not _UNSET and seq is not None:
                raise LaneDispatchError(f'{kind} does not carry a transfer seq')
            if kind != 'layer' and (layer_start is not None or layer_end is not None):
                raise LaneDispatchError(f'{kind} does not carry layer bounds')
        if kind == 'layer':
            if (type(layer_start) is not int or type(layer_end) is not int
                    or not (0 <= layer_start < layer_end)):
                raise LaneDispatchError('layer segment needs half-open global layer bounds')
            if layer_group != 1:
                # Step B: a group names its transaction's size; _do_layer checks that latch.
                if layer_end > layer_start + layer_group:
                    raise LaneDispatchError('layer group is longer than its size')
            elif layer_end != layer_start + 1:
                raise LaneDispatchError('layer segment must cover exactly one layer for now')
        if source is not None:
            from wire_v1 import SourceLeaseDescriptor
            if type(source) is not SourceLeaseDescriptor:
                raise LaneDispatchError('segment source must be a wire source lease')
            if kind != 'prepare':
                raise LaneDispatchError('only prepare carries a source lease')
            if source.transaction != transaction:
                raise LaneDispatchError('source lease belongs to another transaction')
        self.transaction = transaction
        self.stage = stage
        self.kind = kind
        self.op_seq = op_seq
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.seq = None if seq is _UNSET else seq
        self.source = source
        self.profile = profile
        self.layer_group = layer_group

    @classmethod
    def from_descriptor(cls, descriptor):
        """Typed view of a wire descriptor; no int()/str() coercion of its fields."""
        from wire_v1 import SegmentCommand as WireCommand
        if type(descriptor) is WireCommand:
            command, d = descriptor, descriptor.descriptor
            return cls(d.transaction, d.stage, d.kind.value, command.op_seq,
                       layer_start=d.layer_start, layer_end=d.layer_end, seq=d.seq,
                       source=command.source, profile=command.profile, layer_group=d.layer_group)
        kind = descriptor.kind
        if type(kind) is not str:
            raise LaneDispatchError('descriptor kind must be a str')
        layer_start = getattr(descriptor, 'layer_start', None)
        layer_end = getattr(descriptor, 'layer_end', None)
        source = getattr(descriptor, 'source', None)
        if kind in PP_KINDS:
            seq = getattr(descriptor, 'seq', _UNSET)
            if seq is _UNSET:
                raise LaneDispatchError(f'descriptor {kind} needs a transfer seq')
            return cls(descriptor.transaction, descriptor.stage, kind, descriptor.op_seq,
                       layer_start=layer_start, layer_end=layer_end, seq=seq,
                       source=source)
        if getattr(descriptor, 'seq', None) is not None:
            raise LaneDispatchError('non-PP descriptor carries transfer seq')
        return cls(descriptor.transaction, descriptor.stage, kind, descriptor.op_seq,
                   layer_start=layer_start, layer_end=layer_end, source=source)

    def identity(self):
        """Everything that makes two submissions the same accepted command."""
        return (self.transaction, self.stage, self.kind, self.op_seq,
                self.layer_start, self.layer_end, self.seq, self.source, self.profile)


class SegmentEvent:
    __slots__ = ('op_seq', 'kind', 'event', 'host_ns', 'complete')

    def __init__(self, op_seq, kind, event, host_ns):
        self.op_seq = op_seq
        self.kind = kind
        self.event = event
        self.host_ns = host_ns
        self.complete = False


class LaneDispatcher:
    """One worker's lane: admission, segments, completion, cancel and release."""

    def __init__(self, runner, native, factory, *, server=None, qsa_module=None,
                 driver_factory=None, forward_context_getter=None, torch_module=None,
                 clock=time.monotonic_ns):
        self.runner = runner
        self.native = native
        self.factory = factory
        self.server = server
        self.qsa_module = qsa_module
        self.driver_factory = driver_factory
        self.forward_context_getter = forward_context_getter
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.clock = clock
        # Worker-local pinned source ownership outlives one transaction: the
        # bridge closes the opened registry only after release, and a stranded
        # allocation stays visible until then.
        self.source_registry = None
        self.source_allocator = None
        # Held-g device spans (native_measurement.DeviceSpans): installed only
        # with FLASH_S5_NATIVE_PROGRESS set; receipts, never a decision.
        self.spans = None
        self._reset_transaction()

    def _reset_transaction(self):
        """Clear every per-transaction field (open/release share this)."""
        if getattr(self, 'qsa_fused', False):
            self._set_ledger(None)
        self.qsa_fused = False
        self.fused_profiles = None
        self.layer_group = 1
        self.transaction = None
        self.chunk = None
        self.lane = None
        self.driver = None
        self.operation = None
        self.source_binding = None
        self.admission = None
        self.events = []
        self.state_event = None
        self.pp_ticket = None
        self.pp_send_ticket = None
        self.pp_consumer_proof = None
        self._pp_consumer_recorded = False
        self.accepted = {}
        self.cancelled = None
        self.last_ns = None
        self.quarantined = None
        self._partial = False
        self._cancel_owners = ()

    # -- scopes -------------------------------------------------------------
    def _scope(self):
        stack = ExitStack()
        torch = self.torch
        if torch is not None:
            inference = getattr(torch, 'inference_mode', None)
            if inference is not None:
                stack.enter_context(inference())
            stack.enter_context(torch.cuda.stream(self.native.split.prefill_stream))
        else:
            stack.enter_context(nullcontext())
        return stack

    def _record(self, command):
        event = None
        torch = self.torch
        if torch is not None:
            event = torch.cuda.Event(enable_timing=False)
            event.record(self.native.split.prefill_stream)
        record = SegmentEvent(command.op_seq, command.kind, event, self.last_ns)
        self.events.append(record)
        return record

    def _require(self, transaction):
        if self.quarantined is not None:
            raise LaneDispatchError('dispatcher is quarantined: ' + self.quarantined)
        if self.transaction is None or self.transaction != transaction:
            raise LaneDispatchError('no lane transaction matches the command')

    # -- protocol -----------------------------------------------------------
    def open(self, transaction, admission, operation):
        if self.transaction is not None:
            raise LaneDispatchError('a lane transaction is already open on this worker')
        if type(transaction) is not TransactionId:
            raise LaneDispatchError('canonical transaction required')
        if any(not callable(getattr(operation, name, None))
               for name in ('drain', 'cancel', 'release')):
            raise LaneDispatchError('operation must implement drain/cancel/release')
        admission_transaction = getattr(admission, 'transaction', _UNSET)
        if admission_transaction is _UNSET:
            raise LaneDispatchError('open needs the admission transaction identity')
        if admission_transaction != transaction:
            raise LaneDispatchError('admission transaction differs from the lane transaction')
        self._reset_transaction()
        self.transaction = transaction
        self.operation = operation
        self.admission = admission
        if self.spans is not None:
            self.spans.begin(transaction, self._stage())
        return {'opened': True, 'stage': self._stage()}

    # -- Step A: selected at OPEN, before admission; no fallback once selected ----
    def select_fused(self):
        """This open transaction runs each dense QSA layer as one bounded command."""
        if self.transaction is None or self.chunk is not None:
            raise LaneDispatchError('fused QSA is selected at open, before PREPARE')
        self.qsa_fused = True
        self._set_ledger(CollectiveLedger())

    # -- Step B: latched at OPEN with the fused selection -------------------------
    def select_layer_group(self, size):
        """This open transaction runs its LAYER entries as groups of up to ``size``."""
        if self.transaction is None or self.chunk is not None or not self.qsa_fused:
            raise LaneDispatchError('a layer group is selected at open, with the fused selection')
        self.layer_group = size

    def _set_ledger(self, ledger):
        communicator = getattr(self.native.comms, 'tp', None)
        if communicator is not None:
            communicator.ledger = ledger

    def _ledger(self):
        return getattr(getattr(self.native.comms, 'tp', None), 'ledger', None)

    @property
    def stage(self):
        """This worker's PP stage (0 sender, 1 receiver); never inferred by the engine."""
        return self._stage()

    def _stage(self):
        return 0 if not self.native.comms.pp or self.native.comms.pp.is_sender else 1

    def pp_consumer_complete(self):
        """True only when the receiver's consumer event was host-queried complete."""
        ticket = self.pp_ticket
        event = getattr(ticket, 'consumer_event', None)
        return event is not None and bool(event.query())

    def submit(self, command):
        """Run one bounded host operation; returns the accepted record."""
        if not isinstance(command, SegmentCommand):
            command = SegmentCommand.from_descriptor(command)
        self._require(command.transaction)
        stage = self._stage()
        if command.stage != stage:
            raise LaneDispatchError(
                f'segment stage {command.stage} is not this worker stage {stage}')
        existing = self.accepted.get(command.op_seq)
        if existing is not None:
            stored, acknowledgement = existing
            if stored.identity() != command.identity():
                raise LaneDispatchError(
                    f'op_seq {command.op_seq} was accepted for a different command')
            return deepcopy(acknowledgement)
        if self.cancelled is not None:
            raise LaneDispatchError('transaction is cancelled; no new segments')
        if self.accepted and command.op_seq <= max(self.accepted):
            raise LaneDispatchError(
                f'op_seq {command.op_seq} is not strictly increasing')
        started = self.clock()
        self._partial = False
        spans = self.spans
        mark = None
        ledger = self._ledger() if self.qsa_fused else None
        if ledger is not None:
            ledger.begin()
        try:
            with self._scope():
                if spans is not None:
                    # Held-g: the command's start event, before its first launch.
                    mark = spans.start(self.native.split.prefill_stream, self.driver)
                handler = getattr(self, '_do_' + command.kind)
                result = handler(command)
            self.last_ns = self.clock() - started
            # The handler has enqueued its work; recording is now part of the
            # same partial-enqueue exposure.
            self._partial = True
            record = self._record(command)
            if spans is not None:
                spans.end(mark, self.native.split.prefill_stream, command, self.driver)
            self._after_record(command, record)
        except E3Pending as exc:
            if spans is not None:
                spans.refused(None if self._partial else mark, command, exc)
            # Known credit checks occur before submission. Other pending
            # exceptions after mutation have an unknown enqueue frontier.
            if self._partial:
                self.quarantined = command.kind + ': pending after possible enqueue'
            raise
        except BaseException as exc:
            if spans is not None:
                spans.abandon(mark)
            if self._partial:
                self.quarantined = f'{command.kind}: {type(exc).__name__}'
                raise LaneDispatchError('dispatcher quarantined: ' + self.quarantined) from exc
            raise
        acknowledgement = {'op_seq': command.op_seq, 'kind': command.kind, 'result': result,
                            'cursor': self.cursor(), 'host_submit_ns': self.last_ns}
        if self.qsa_fused:
            acknowledgement['collectives'] = ledger.take() if ledger is not None else (0, '')
        self.accepted[command.op_seq] = (deepcopy(command), deepcopy(acknowledgement))
        return deepcopy(acknowledgement)

    def _after_record(self, command, record):
        """Conservative PP consumer boundary: all layer tails and state writes finished."""
        if (command.kind == 'state_finalize' and self._stage() == 1
                and self.pp_ticket is not None and not self._pp_consumer_recorded):
            event = record.event
            self.native.comms.pp.consumer_done(self.pp_ticket, event)
            self._pp_consumer_recorded = True

    # -- handlers -------------------------------------------------------------
    def _do_prepare(self, command):
        if self.chunk is not None:
            raise LaneDispatchError('chunk already prepared for this transaction')
        if command.source is not None:
            binding = self.source_binding
            if binding is None or binding.descriptor != command.source:
                raise LaneDispatchError(
                    'prepare source lease was not bound for this command')
        self._partial = True
        chunk = self.factory.admit(self.admission)
        self.chunk = chunk
        driver = None
        if self.driver_factory is not None:
            driver = self.driver_factory(self.operation, self.qsa_module,
                                         self.forward_context_getter)
        self.driver = driver
        self._partial = True
        self.lane = self.native.lane_forward(chunk.lane_inputs(), forward_context=chunk.forward_scope,
                                             attention_driver=driver)
        if self.server is not None:
            marker = type('Parked', (), {})()
            marker.flash_e3_transaction = command.transaction
            # The bound source lease reaches the server through this marker;
            # this dispatch never invents a lease of its own.
            marker.flash_e3_source_binding = self.source_binding
            self.server.begin_prefill_operation(self.runner, marker)
        if self.qsa_fused:
            # Host plan only: the peers agree on these before the stage's first collective.
            if driver is None:
                raise LaneDispatchError('fused QSA needs the constructed layer driver')
            self.fused_profiles = driver.fused_profiles(self.lane, self.admission.chunk_end)
            return {'prepared': True, 'req_index': chunk.req_index,
                    'dense_profiles': self.fused_profiles}
        return {'prepared': True, 'req_index': chunk.req_index}

    def _do_pp_receive(self, command):
        channel = self.native.comms.pp
        if channel is None or channel.is_sender:
            raise LaneDispatchError('pp_receive is only valid on the receiving stage')
        if self.pp_ticket is not None:
            raise LaneDispatchError('PP receive already posted')
        self._partial = True
        self.pp_ticket = channel.post_receive(command.transaction, command.stage - 1, command.seq,
                                              self.admission.num_tokens)
        return {'posted': True}

    def _do_prefix(self, command):
        self._need_chunk()
        intermediate = None
        if self._stage() == 1:
            if self.pp_ticket is None:
                raise LaneDispatchError('stage 1 prefix needs a posted PP receive')
            # A query, but a channel error here means the receive state is
            # untrustworthy; treat the scope as partially exposed.
            self._partial = True
            payload = self.native.comms.pp.poll_receive(self.pp_ticket)
            if payload is None:
                self._partial = False
                raise E3Pending('PP receive has not completed')
            intermediate = {'hidden_states': payload}
        self._partial = True
        record = self.lane.begin(intermediate_tensors=intermediate)
        return {'kind': record.kind}

    def _do_layer(self, command):
        self._need_chunk()
        if command.layer_start != self.lane.cursor:
            raise LaneDispatchError(f'layer segment {command.layer_start} does not match cursor {self.lane.cursor}')
        if command.layer_group != self.layer_group:
            raise LaneDispatchError('layer group differs from this transaction')
        if self.layer_group != 1:
            return self._do_layer_group(command)
        if self.qsa_fused:
            return self._do_layer_fused(command)
        # Query the driver's source-owned credit *before* the partial-enqueue
        # boundary: a credit-only attempt must stay typed pending for the
        # unchanged command and must record no event or accepted operation.
        query = getattr(self.driver, 'submission_credit', None)
        credit = query(command.layer_start) if callable(query) else None
        if credit is not None and getattr(credit, 'can_submit', None) is not True:
            raise E3Pending(
                'lane submission credit pending: '
                + str(getattr(credit, 'reason', 'unspecified')))
        authorize = getattr(self.driver, 'authorize_profile', None)
        if callable(authorize):
            authorize(command.layer_start, command.profile)
        elif command.profile is not None:
            raise LaneDispatchError('QSA profile grant without its constructed driver')
        self._partial = True
        record = self.lane.layer(command.layer_start)
        boundary = getattr(self.driver, 'at_schedule_boundary', None)
        tail_boundary = getattr(self.driver, 'at_tail_boundary', None)
        return {'kind': record.kind, 'layer': command.layer_start,
                'paused': record.kind == 'layer-paused',
                'profile_boundary': bool(callable(boundary) and boundary(command.layer_start)),
                'tail_boundary': bool(callable(tail_boundary) and tail_boundary(command.layer_start))}

    def _do_layer_fused(self, command):
        """A QSA layer of a fused transaction: its grant or nothing, never the stepped path."""
        grant = command.profile
        owned = dict(self.fused_profiles).get(command.layer_start)
        if grant is None:
            if owned is not None:
                raise LaneDispatchError('a fused transaction refuses a stepped QSA layer')
            self._partial = True
            record = self.lane.layer(command.layer_start)
        else:
            if grant.fused is not True or owned != grant.shape:
                raise LaneDispatchError('fused grant differs from this PREPARE plan')
            self.driver.arm_fused(command.layer_start, grant)
            self._partial = True
            record = self.lane.layer(command.layer_start)
        if record.kind != 'layer':
            raise LaneDispatchError('a fused transaction layer did not finish in its command')
        return {'kind': record.kind, 'layer': command.layer_start, 'paused': False,
                'profile_boundary': False, 'tail_boundary': False}

    def _do_layer_group(self, command):
        """Step B: consecutive layers of this stage as one bounded command.

        Everything is refused before the first enqueue. From the first layer on, anything but
        a finished layer quarantines: the peer may be inside a later collective already.
        """
        start, end, grant = command.layer_start, command.layer_end, command.profile
        owned = dict(self.fused_profiles)
        if end > self.lane.end_layer:
            raise LaneDispatchError('a layer group crosses its stage')
        qsa = [layer for layer in range(start, end) if layer in owned]
        if qsa not in ([], [end - 1]):
            raise LaneDispatchError('a layer group holds its QSA layer last, and only one')
        if (grant is None) != (not qsa):
            raise LaneDispatchError('a layer group carries the grant of its QSA layer, or none')
        if grant is not None and (grant.fused is not True or owned[end - 1] != grant.shape):
            raise LaneDispatchError('fused grant differs from this PREPARE plan')
        self._partial = True
        for layer in range(start, end):
            if layer in owned:
                self.driver.arm_fused(layer, grant)
            record = self.lane.layer(layer)
            if record.kind != 'layer':
                raise LaneDispatchError('a layer group layer did not finish in its command')
        return {'kind': record.kind, 'layer': end - 1, 'paused': False,
                'profile_boundary': False, 'tail_boundary': False}

    def _do_suffix(self, command):
        self._need_chunk()
        self._partial = True
        record = self.lane.finish()
        return {'kind': record.kind, 'has_output': self.lane.output is not None}

    def _do_pp_send(self, command):
        self._need_chunk()
        channel = self.native.comms.pp
        if channel is None or not channel.is_sender:
            raise LaneDispatchError('pp_send is only valid on the sending stage')
        if self.lane.output is None:
            raise LaneDispatchError('pp_send before the suffix produced the HC tensor')
        if self.pp_send_ticket is not None:
            raise LaneDispatchError('PP send already enqueued')
        self._partial = True
        self.pp_send_ticket = channel.send(self.lane.output['hidden_states'], command.transaction,
                                           command.stage, command.seq)
        return {'sent': True}

    def _do_state_finalize(self, command):
        self._need_chunk()
        if self.lane.state != 'finished':
            raise LaneDispatchError('state_finalize before the suffix')
        if self.state_event is not None:
            raise LaneDispatchError('state already finalized')
        if self.native.comms.pp is not None:
            ticket = self.pp_send_ticket if self._stage() == 0 else self.pp_ticket
            if ticket is None:
                raise LaneDispatchError('state finalize requires the stage PP transfer')
        self._partial = True
        self.chunk.finish_state()
        torch = self.torch
        if torch is not None:
            copy_stream = getattr(self.native, 'copy_stream', None)
            if copy_stream is not None:
                join = torch.cuda.Event(enable_timing=False)
                join.record(copy_stream)
                self.native.split.prefill_stream.wait_event(join)
            event = torch.cuda.Event(enable_timing=False)
            event.record(self.native.split.prefill_stream)
            self.state_event = event
        else:
            raise LaneDispatchError('CUDA event provider missing')
        if self.server is not None:
            self.server.end_prefill_operation(self.runner)
        return {'finalized': True}

    def _need_chunk(self):
        if self.chunk is None or self.lane is None:
            raise LaneDispatchError('segment before prepare')

    # -- completion -------------------------------------------------------------
    def cursor(self):
        if self.lane is None:
            return None
        inner = self.driver.cursor() if self.driver is not None else None
        return (self.lane.cursor, self.lane.state, inner)

    def poll(self):
        """Host-queried completion; failed queries quarantine retained owners."""
        try:
            return self._poll()
        except BaseException as exc:
            self.quarantined = 'completion query: ' + type(exc).__name__
            raise

    def _poll(self):
        """Never waits or advances a model continuation."""
        completed = []
        pending = []
        for record in self.events:
            if not record.complete and record.event is not None and bool(record.event.query()):
                record.complete = True
            (completed if record.complete else pending).append(record.op_seq)
        state_written = self.state_event is not None and (
            bool(self.state_event.query()))
        pp_ready = None
        if self.pp_ticket is not None:
            channel = self.native.comms.pp
            if self.quarantined is not None or channel is None:
                pp_ready = bool(getattr(self.pp_ticket, 'state', None) == 'ready')
            else:
                try:
                    payload = channel.poll_receive(self.pp_ticket)
                except LaneCommError as exc:
                    self.quarantined = 'PP receive poll: ' + type(exc).__name__
                    pp_ready = False
                else:
                    pp_ready = (payload is not None
                                or getattr(self.pp_ticket, 'state', None) == 'ready')
        return {'completed_op_seqs': tuple(completed), 'pending_op_seqs': tuple(pending),
                'state_written': state_written, 'pp_receive_ready': pp_ready,
                'cursor': self.cursor(), 'cancelled': self.cancelled,
                'quarantined': self.quarantined}

    def cancel(self, reason='cancelled'):
        if self.transaction is None:
            raise LaneDispatchError('no transaction to cancel')
        if self.cancelled is not None:
            return self.cancelled
        self.cancelled = reason
        # abort helpers may clear cursors; keep the actual tensor owners until
        # all submitted work is queried complete.
        self._cancel_owners = (self.chunk,
            tuple(getattr(self.lane, name, None) for name in
                  ('hidden_states', 'block_output', 'injection', 'output')),
            dict(getattr(self.driver, 'phases', {})))
        self.operation.cancel(reason)
        if self.lane is not None:
            self.lane.abort(reason)
        if self.driver is not None:
            self.driver.abort(reason)
        if self.server is not None and self.chunk is not None:
            self.server.abort_prefill_operation(self.runner)
        return reason

    def accept_pp_consumer_proof(self, proof):
        """Store the engine's PP_CONSUMER proof after identity validation."""
        if self.quarantined is not None:
            raise LaneDispatchError('dispatcher is quarantined: ' + self.quarantined)
        if self.transaction is None:
            raise LaneDispatchError('no lane transaction is open')
        if self.pp_send_ticket is None:
            raise LaneDispatchError('no PP send is outstanding')
        self.native.comms.pp.validate_consumer_proof(self.pp_send_ticket, proof)
        self.pp_consumer_proof = proof
        return True

    def drain(self):
        if self.quarantined is not None:
            return False
        report = self.poll()
        if self.quarantined is not None or report['pending_op_seqs']:
            return False
        operation = self.operation
        if operation is None or not operation.drain():
            return False
        if self.cancelled is None and (self.state_event is None or not report['state_written']):
            return False
        if self.pp_ticket is not None:
            # Cancellation before the normal final consumer: once every
            # accepted command and receive has completed, no consumer remains.
            if (self.cancelled is not None and report['pp_receive_ready']
                    and not self._pp_consumer_recorded):
                self.native.comms.pp.consumer_done(self.pp_ticket, self.pp_ticket.event)
                self._pp_consumer_recorded = True
            consumer = getattr(self.pp_ticket, 'consumer_event', None)
            if consumer is None or not bool(consumer.query()):
                return False
        if self.pp_send_ticket is not None:
            if self.pp_consumer_proof is None:
                return False
            event = getattr(self.pp_send_ticket, 'event', None)
            if event is None or not bool(event.query()):
                return False
        return True

    def release(self):
        """Drop the transaction after drain; the request row stays for the next chunk."""
        if self.quarantined is not None:
            raise LaneDispatchError('dispatcher is quarantined: ' + self.quarantined)
        if not self.drain():
            raise E3Pending('lane transaction still has outstanding work')
        if self.spans is not None:
            # Held-g: every command event is complete; read and log the spans.
            self.spans.release(self.native.split.prefill_stream, self.cancelled)
        try:
            channel = self.native.comms.pp
            if channel is not None:
                if self.pp_ticket is not None:
                    channel.release_receive(self.pp_ticket)
                if self.pp_send_ticket is not None:
                    channel.release_send(self.pp_send_ticket, self.pp_consumer_proof)
            self.operation.release()
            if self.chunk is not None:
                # drain() above queried every segment/state/PP consumer event
                # and operation.drain() proved its internal readers complete.
                self.factory.complete_chunk(self.chunk, consumers_complete=True)
        except BaseException as exc:
            self.quarantined = 'release: ' + type(exc).__name__
            raise
        self._reset_transaction()
        return True

    def close_request(self):
        """The request left the lane (handback or abort): free its worker row."""
        if self.transaction is not None:
            raise LaneDispatchError('close_request while a transaction is open')
        return self.factory.release()


__all__ = ['KINDS', 'LaneDispatchError', 'LaneDispatcher', 'SegmentCommand', 'SegmentEvent']
