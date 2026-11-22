"""E3 native backends: device events, split-graph capture, engine-side relay.

This module wraps E3's proposed torch operations.  torch is imported
lazily inside :class:`NativeDevice` so everything else stays importable on the
CPU hub.  The classes here are constructed by real startup code:

* :class:`EventCompletionBackend` on each GPU worker, from
  ``native_resources.build_worker_runtime`` (called by the generated runner
  before A's ``init_lane``).
* :class:`NativeGraphCaptureBackend` on each GPU worker, from
  ``native_resources.install_graph_builder`` (called by B's generated
  ``capture_banks`` seam).
* :class:`RemoteCompletionBackend` on the engine, from
  ``s5_startup.install_engine`` (called by the generated ``EngineCore.__init__``
  seam).  It owns no device and never produces a proof: worker proofs travel
  back as typed poll replies.

Evidence boundary: a completion proof here is a device event recorded on the
stream after the work was *launched*.  It proves the stream reached that point;
it does not prove which kernels ran, that a bounded segment (rather than the
whole stock forward) ran, or that the PP consumer read its buffer.  Those remain
owner-run checks.  ``native_verified`` is a capability declaration of the
implementation. The generic worker-event adapter remains unverified for
state/PP completion. The engine relay can declare implementation capability
only with its exclusively owned reply transport and verified package closure;
that declaration is not a GPU or release result.
"""
import time

from async_handoff import CompletionBackend, ConfigurationError, HandoffConfig
from reply_transport import ReplyTransport
from e3_types import (
    CompletionKind,
    CompletionProof,
    DependencyRole,
    E3ContractError,
    E3Pending,
)


class NativeBackendError(E3ContractError):
    pass


# Role -> proof kind.  A producer/copy/attention operation completes on the
# compute stream. A generic event is NOT a state-write or PP-consumer proof;
# those roles require native operation-specific instrumentation not implemented here.
_ROLE_KIND = {
    DependencyRole.PRODUCER: CompletionKind.GPU_EVENT,
    DependencyRole.CONSUMER: CompletionKind.GPU_EVENT,
}


def _torch():
    import torch  # Local import: the hub has no CUDA runtime.

    return torch


class NativeDevice:
    """Thin, explicit wrapper over the CUDA calls E3 startup needs."""

    def __init__(self, torch_module=None, device=None):
        self.torch = torch_module if torch_module is not None else _torch()
        if device is None:
            device = self.torch.device('cuda',
                                       self.torch.cuda.current_device())
        self.device = device

    # -- allocation -------------------------------------------------------
    def allocate(self, nbytes, *, name):
        if type(nbytes) is not int or nbytes <= 0:
            raise NativeBackendError('positive allocation size required')
        tensor = self.torch.empty((nbytes,), dtype=self.torch.uint8,
                                  device=self.device)
        return DeviceBuffer(name, tensor)

    def allocate_i32(self, length, *, name, fill=-2):
        if type(length) is not int or length <= 0:
            raise NativeBackendError('positive buffer length required')
        tensor = self.torch.full((length,), fill, dtype=self.torch.int32,
                                 device=self.device)
        return DeviceBuffer(name, tensor)

    # -- events -----------------------------------------------------------
    def event(self):
        return self.torch.cuda.Event(enable_timing=False)

    def record(self, event, stream=None):
        event.record(stream)

    def query(self, event):
        return bool(event.query())

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def current_stream_token(self):
        stream = self.torch.cuda.current_stream(self.device)
        token = getattr(stream, 'cuda_stream', None)
        if type(token) is not int:
            raise NativeBackendError('current stream has no device token')
        return token

    def sm_count(self):
        props = self.torch.cuda.get_device_properties(self.device)
        count = getattr(props, 'multi_processor_count', None)
        if type(count) is not int or count < 1:
            raise NativeBackendError('device reported no measured SM count')
        return count

    def memory_allocated(self):
        return int(self.torch.cuda.memory_allocated(self.device))


class DeviceBuffer:
    """One owned allocation; only the address is handed to A/B."""

    def __init__(self, name, tensor):
        if not isinstance(name, str) or not name:
            raise NativeBackendError('buffer name required')
        self.name = name
        self.tensor = tensor
        self.nbytes = int(tensor.numel() * tensor.element_size())

    @property
    def address(self):
        return int(self.tensor.data_ptr())

    def __repr__(self):
        return f'DeviceBuffer({self.name}, {self.nbytes} bytes)'


class EventCompletionBackend(CompletionBackend):
    """Device-event completion source owned by one GPU worker.

    ``submit`` registers the ticket; ``record_launch`` records the CUDA event
    after the work was enqueued.  ``poll`` only queries that event and never
    synchronizes.  A ticket without a recorded event (e.g. a submission whose
    forward has not been launched yet) stays pending rather than completing.
    """

    native_verified = False
    fixture_only = False

    def __init__(self, device, *, clock=time.monotonic):
        if not isinstance(device, NativeDevice):
            raise ConfigurationError('event backend needs the native device')
        self.device = device
        self.clock = clock
        self._pending = {}
        self._closed = False

    def submit(self, ticket):
        if self._closed:
            raise NativeBackendError('completion backend is closed')
        role = ticket.dependency_role
        if role not in _ROLE_KIND:
            raise ConfigurationError(
                'state/PP-consumer completions are not instrumented; COMMIT/HANDOFF operations '
                'cannot be submitted to the event backend')
        if ticket in self._pending:
            raise NativeBackendError('operation submitted twice')
        self._pending[ticket] = {'event': None, 'kind': _ROLE_KIND[role],
                                 'failed': None}
        return ticket

    def record_launch(self, ticket):
        """Record the device event for a ticket whose work was just enqueued."""
        if self._closed:
            raise NativeBackendError('completion backend is closed')
        entry = self._pending.get(ticket)
        if entry is None:
            raise NativeBackendError('launch recorded for an unknown operation')
        if entry['failed'] is not None:
            raise NativeBackendError('launch recorded for a failed operation')
        if entry['event'] is not None:
            raise NativeBackendError('launch recorded twice for one operation')
        event = self.device.event()
        self.device.record(event)
        entry['event'] = event
        return event

    def fail(self, ticket, reason='device failure'):
        entry = self._pending.get(ticket)
        if entry is None:
            raise NativeBackendError('failed operation is not outstanding')
        entry['failed'] = reason or 'device failure'

    def poll(self, transaction):
        ready = []
        for ticket, entry in list(self._pending.items()):
            if ticket.transaction != transaction:
                continue
            if entry['failed'] is not None:
                raise NativeBackendError(entry['failed'])
            if entry['event'] is None or not self.device.query(entry['event']):
                continue
            del self._pending[ticket]
            # Equal on every rank for the same operation and monotonic per lane.
            ready.append(CompletionProof(ticket, entry['kind'],
                                         ticket.op_seq + 1))
        return tuple(ready)

    def outstanding(self, transaction):
        return tuple(ticket for ticket in self._pending
                     if ticket.transaction == transaction)

    def close(self):
        self._closed = True
        return dict(pending=len(self._pending))


class RemoteCompletionBackend(CompletionBackend):
    """Engine-side relay declaration; it owns no device and completes nothing.

    Worker proofs arrive as typed poll replies and are applied by
    :class:`async_handoff.HandoffDispatcher`.  This object exists so enabled
    engine startup rejects the fixture backend explicitly instead of silently
    running with no completion source.
    """

    fixture_only = False
    engine_relay_only = True

    def __init__(self, config, *, transport=None):
        if not isinstance(config, HandoffConfig):
            raise ConfigurationError('relay requires the validated HandoffConfig')
        self.config = config
        self._transport = transport
        if transport is not None:
            self._check_transport()

    def _check_transport(self):
        transport = self._transport
        if not self.config.enabled or not isinstance(transport, ReplyTransport):
            raise ConfigurationError('native relay requires the enabled owned ReplyTransport')
        transport._owner()
        with transport._condition:
            transport._healthy()
            ranks = tuple(transport._queues)
            if (ranks != tuple(range(len(ranks))) or not ranks
                    or not 0 < transport._max_calls <= self.config.command_slots
                    or transport._status is None
                    or any(getattr(queue, '_flash_s5_receiver_owner', None) is not transport
                           for queue in transport._queues.values())):
                raise ConfigurationError('native relay reply ownership or capacity is unproved')

    @property
    def native_verified(self):
        # No naked flag or caller-provided ready dict can promote this relay.
        # The source/build guard is the same real package-local check used by
        # startup. It certifies implementation only; worker bootstrap, typed
        # queried event proofs and same-queue hardware gates remain required.
        if self._transport is None:
            return False
        self._check_transport()
        from native_readiness import require_implementation_closure
        capability = require_implementation_closure()
        return (capability.get('native_ready') is True
                and capability.get('hardware_qualified') is False
                and capability.get('release_qualified') is False)

    def submit(self, ticket):
        raise NativeBackendError(
            'the engine never submits device work through the relay backend')

    def poll(self, transaction):
        return ()


class GraphHandle:
    """Opaque token returned by :meth:`NativeGraphCaptureBackend.capture`."""

    def __init__(self, graph, output, buffers, key, stream):
        self.graph = graph
        self.output = output
        self.buffers = tuple(buffers)
        self.key = key
        self.stream = stream
        self.replays = 0
        self.destroyed = False


class NativeGraphCaptureBackend:
    """torch.cuda graph capture/replay for B's declared split bank.

    ``capture_callable(descriptor, key)`` is the model-forward seam.  It is
    supplied by the worker integration; when it is absent startup fails closed
    instead of registering an empty bank that could later route full-GPU decode
    during a parked prefill.
    """

    def __init__(self, device, capture_callable=None, *, clock=time.monotonic):
        if not isinstance(device, NativeDevice):
            raise ConfigurationError('graph backend needs the native device')
        self.device = device
        self.capture_callable = capture_callable
        self.clock = clock
        self._events = set()
        self._streams = set()

    def capture(self, bank_name, key, lane, context):
        if not callable(self.capture_callable):
            raise ConfigurationError(
                'native split capture needs the model-forward capture callable '
                '(runner._flash_s5_graph_capture); it is not wired')
        raise ConfigurationError('native green context, measured graph pool and stream binding '
                                 'are not implemented; ordinary torch.cuda.graph is not split capture')

    def replay(self, handle, key, lane, context):
        if not isinstance(handle, GraphHandle):
            raise NativeBackendError('replay needs a captured graph handle')
        if handle.destroyed:
            raise NativeBackendError('replay of a destroyed graph')
        if handle.key != key:
            raise NativeBackendError('replay uses a different graph key')
        handle.graph.replay()
        handle.replays += 1
        return handle.output

    def destroy_graph(self, handle):
        if not isinstance(handle, GraphHandle):
            raise NativeBackendError('destroy needs a captured graph handle')
        if handle.destroyed:
            raise NativeBackendError('graph already destroyed')
        handle.destroyed = True
        handle.graph = None
        handle.output = None
        handle.buffers = ()

    def destroy_event(self, token):
        if id(token) not in self._events:
            raise NativeBackendError('destroy of an unregistered event')
        self._events.discard(id(token))

    def destroy_stream(self, token):
        if id(token) not in self._streams:
            raise NativeBackendError('destroy of an unregistered stream')
        self._streams.discard(id(token))


__all__ = [
    'DeviceBuffer', 'EventCompletionBackend', 'GraphHandle',
    'NativeBackendError', 'NativeDevice', 'NativeGraphCaptureBackend',
    'RemoteCompletionBackend',
]
