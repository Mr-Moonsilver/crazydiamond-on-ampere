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
"""
import time
from contextlib import ExitStack, nullcontext

from e3_types import E3ContractError, E3Pending

KINDS = ('prepare', 'pp_receive', 'prefix', 'layer', 'suffix', 'pp_send', 'state_finalize')


class LaneDispatchError(E3ContractError):
    pass


class SegmentCommand:
    """Local view of one wire-v1 SegmentDescriptor."""

    __slots__ = ('transaction', 'stage', 'layer_start', 'layer_end', 'op_seq', 'kind', 'seq')

    def __init__(self, transaction, stage, kind, op_seq, *, layer_start=None, layer_end=None, seq=0):
        if kind not in KINDS:
            raise LaneDispatchError(f'unknown segment kind {kind!r}')
        for name, value in (('stage', stage), ('op_seq', op_seq), ('seq', seq)):
            if type(value) is not int or value < 0:
                raise LaneDispatchError(f'segment {name} must be a non-negative int')
        if kind == 'layer':
            if type(layer_start) is not int or type(layer_end) is not int or not (0 <= layer_start < layer_end):
                raise LaneDispatchError('layer segment needs half-open global layer bounds')
        self.transaction = transaction
        self.stage = stage
        self.kind = kind
        self.op_seq = op_seq
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.seq = seq

    @classmethod
    def from_descriptor(cls, descriptor):
        return cls(descriptor.transaction, int(descriptor.stage), str(descriptor.kind),
                   int(descriptor.op_seq), layer_start=getattr(descriptor, 'layer_start', None),
                   layer_end=getattr(descriptor, 'layer_end', None),
                   seq=int(getattr(descriptor, 'seq', 0) or 0))


class SegmentEvent:
    __slots__ = ('op_seq', 'kind', 'event', 'host_ns', 'complete')

    def __init__(self, op_seq, kind, event, host_ns):
        self.op_seq = op_seq
        self.kind = kind
        self.event = event
        self.host_ns = host_ns
        self.complete = event is None


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
        self.torch = torch_module
        self.clock = clock
        self.transaction = None
        self.chunk = None
        self.lane = None
        self.driver = None
        self.operation = None
        self.events = []
        self.state_event = None
        self.pp_ticket = None
        self.pp_send_ticket = None
        self.accepted = []
        self.cancelled = None
        self.last_ns = None

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
        if self.transaction is None or self.transaction != transaction:
            raise LaneDispatchError('no lane transaction matches the command')
        if self.cancelled is not None:
            raise LaneDispatchError('transaction is cancelled; no new segments')

    # -- protocol -----------------------------------------------------------
    def open(self, transaction, admission, operation):
        if self.transaction is not None:
            raise LaneDispatchError('a lane transaction is already open on this worker')
        if operation is None:
            raise LaneDispatchError('open needs the worker QSA operation')
        self.transaction = transaction
        self.operation = operation
        self.admission = admission
        self.cancelled = None
        self.events = []
        self.state_event = None
        self.accepted = []
        return {'opened': True, 'stage': self._stage()}

    def _stage(self):
        return 0 if not self.native.comms.pp or self.native.comms.pp.is_sender else 1

    def submit(self, command):
        """Run one bounded host operation; returns the accepted record."""
        if not isinstance(command, SegmentCommand):
            command = SegmentCommand.from_descriptor(command)
        self._require(command.transaction)
        started = self.clock()
        with self._scope():
            handler = getattr(self, '_do_' + command.kind)
            result = handler(command)
        self.last_ns = self.clock() - started
        record = self._record(command)
        self.accepted.append((command.op_seq, self.last_ns))
        return {'op_seq': command.op_seq, 'kind': command.kind, 'result': result,
                'cursor': self.cursor(), 'host_submit_ns': self.last_ns}

    # -- handlers -------------------------------------------------------------
    def _do_prepare(self, command):
        if self.chunk is not None:
            raise LaneDispatchError('chunk already prepared for this transaction')
        chunk = self.factory.admit(self.admission)
        self.chunk = chunk
        driver = None
        if self.driver_factory is not None:
            driver = self.driver_factory(self.operation, self.qsa_module,
                                         self.forward_context_getter)
        self.driver = driver
        self.lane = self.native.lane_forward(chunk.lane_inputs(), forward_context=chunk.forward_scope,
                                             attention_driver=driver)
        if self.server is not None:
            marker = type('Parked', (), {})()
            marker.flash_e3_transaction = command.transaction
            self.server.begin_prefill_operation(self.runner, marker)
        return {'prepared': True, 'req_index': chunk.req_index}

    def _do_pp_receive(self, command):
        channel = self.native.comms.pp
        if channel is None or channel.is_sender:
            raise LaneDispatchError('pp_receive is only valid on the receiving stage')
        if self.pp_ticket is not None:
            raise LaneDispatchError('PP receive already posted')
        self.pp_ticket = channel.post_receive(command.transaction, command.stage - 1, command.seq,
                                              self.admission.num_tokens)
        return {'posted': True}

    def _do_prefix(self, command):
        self._need_chunk()
        intermediate = None
        if self._stage() == 1:
            if self.pp_ticket is None:
                raise LaneDispatchError('stage 1 prefix needs a posted PP receive')
            payload = self.native.comms.pp.poll_receive(self.pp_ticket)
            if payload is None:
                raise E3Pending('PP receive has not completed')
            intermediate = {'hidden_states': payload}
        record = self.lane.begin(intermediate_tensors=intermediate)
        return {'kind': record.kind}

    def _do_layer(self, command):
        self._need_chunk()
        if command.layer_start != self.lane.cursor:
            raise LaneDispatchError(f'layer segment {command.layer_start} does not match cursor {self.lane.cursor}')
        record = self.lane.layer(command.layer_start)
        return {'kind': record.kind, 'layer': command.layer_start, 'paused': record.kind == 'layer-paused'}

    def _do_suffix(self, command):
        self._need_chunk()
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
        self.pp_send_ticket = channel.send(self.lane.output['hidden_states'], command.transaction,
                                           command.stage, command.seq)
        return {'sent': True}

    def _do_state_finalize(self, command):
        self._need_chunk()
        if self.lane.state != 'finished':
            raise LaneDispatchError('state_finalize before the suffix')
        if self.state_event is not None:
            raise LaneDispatchError('state already finalized')
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
            self.state_event = 'state-event'
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
        """Host-queried completion; never waits."""
        completed = []
        pending = []
        for record in self.events:
            if not record.complete and (record.event is None or bool(record.event.query())):
                record.complete = True
            (completed if record.complete else pending).append(record.op_seq)
        state_written = self.state_event is not None and (
            self.state_event == 'state-event' or bool(self.state_event.query()))
        pp_ready = None
        if self.pp_ticket is not None:
            pp_ready = self.pp_ticket.state == 'ready'
        return {'completed_op_seqs': tuple(completed), 'pending_op_seqs': tuple(pending),
                'state_written': state_written, 'pp_receive_ready': pp_ready,
                'cursor': self.cursor(), 'cancelled': self.cancelled}

    def cancel(self, reason='cancelled'):
        if self.transaction is None:
            raise LaneDispatchError('no transaction to cancel')
        self.cancelled = reason
        if self.lane is not None:
            self.lane.abort(reason)
        if self.driver is not None:
            self.driver.abort(reason)
        if self.server is not None and self.chunk is not None:
            self.server.abort_prefill_operation(self.runner)
        return reason

    def drain(self):
        report = self.poll()
        if report['pending_op_seqs']:
            return False
        if self.cancelled is None and self.state_event is not None and not report['state_written']:
            return False
        return True

    def release(self):
        """Drop the transaction after drain; the request row stays for the next chunk."""
        if not self.drain():
            raise E3Pending('lane transaction still has outstanding work')
        self.transaction = None
        self.chunk = None
        self.lane = None
        self.driver = None
        self.operation = None
        self.pp_ticket = None
        self.pp_send_ticket = None
        self.state_event = None
        return True

    def close_request(self):
        """The request left the lane (handback or abort): free its worker row."""
        if self.transaction is not None:
            raise LaneDispatchError('close_request while a transaction is open')
        return self.factory.release()


__all__ = ['KINDS', 'LaneDispatchError', 'LaneDispatcher', 'SegmentCommand', 'SegmentEvent']
