"""C: executor/core/worker source seams for the async prefill handoff.

The transforms here are narrow, anchored edits of the pinned vLLM sources:

* ``v1/executor/multiproc_executor.py`` -- every ``collective_rpc`` reply is
  routed through one owned receiver at an idle handoff. The stock
  ``FutureWrapper`` route stays byte-identical when overlap is off and is
  explicitly unreachable, not merely unused, when the transport is installed.
* ``v1/engine/core.py`` -- ``step_with_batch_queue`` gets the bounded engine
  bypass; abort/pause/reset/shutdown stop submissions before release.
* ``v1/worker/gpu_worker.py`` -- typed S5 RPC entry points delegate to C's
  worker adapter. ACK, completion and publication stay separate.

No vLLM import happens at module import time, so the transform and its CPU
fixtures run offline. Native shared-memory/ZMQ, CUDA reply deserialization, PP
transport and real engine timing are owner-run acceptance steps.
"""
import hashlib
import math
import time
import threading
import os
import json
import logging
import sys

from e3_types import E3ContractError, E3Pending
from reply_transport import Backpressure, NotReady, ReplyTransport, TransportError, WorkerRPCError


EXECUTOR_PATH = 'v1/executor/multiproc_executor.py'
CORE_PATH = 'v1/engine/core.py'
WORKER_PATH = 'v1/worker/gpu_worker.py'

EXECUTOR_SHA256 = '3ed0149a8a8e1550c4f062af40e645d1b734e70f99981bc5bf15f0360507648a'
CORE_SHA256 = '268bf40534a853812867ae8e80040f190983cb364b41182b55969aace20c93f5'
WORKER_SHA256 = '596f5b6e97be754f84b04c17bc0b96fd92d2f05ee86e0459aa515ffa48d8c5cb'

PINS = {EXECUTOR_PATH: EXECUTOR_SHA256,
        CORE_PATH: CORE_SHA256,
        WORKER_PATH: WORKER_SHA256}

_IMPORT = '''import vllm.v1.flash_e3.executor_hooks as _flash_s5_hooks
'''

_EXECUTOR_WRAPPER = '''    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator | None = None,
        ec_output_aggregator: ECOutputAggregator | None = None,
    ) -> Any:
        return _flash_s5_hooks.collective_rpc(
            self,
            self._flash_s5_stock_collective_rpc,
            method,
            timeout,
            args,
            kwargs,
            non_block,
            unique_reply_rank,
            kv_output_aggregator,
            ec_output_aggregator,
        )

'''

_EXECUTOR_GUARD = '''        if getattr(self, "_flash_s5_stock_forbidden", False):
            raise RuntimeError(
                "stock collective_rpc response reader is disabled while the "
                "E3 owned receiver is installed"
            )
'''

_WORKER_METHODS = '''    def flash_s5_open(self, message):
        return _flash_s5_hooks.worker_rpc(self, "open", message)

    def flash_s5_submit(self, message):
        return _flash_s5_hooks.worker_rpc(self, "submit", message)

    def flash_s5_poll(self, message):
        return _flash_s5_hooks.worker_rpc(self, "poll", message)

    def flash_s5_commit(self, message):
        return _flash_s5_hooks.worker_rpc(self, "commit", message)

    def flash_s5_release(self, message):
        return _flash_s5_hooks.worker_rpc(self, "release", message)

    def flash_s5_bind(self, message):
        return _flash_s5_hooks.worker_rpc(self, "bind", message)

    def flash_s5_receipt(self, message):
        return _flash_s5_hooks.worker_rpc(self, "receipt", message)

    def flash_s5_cancel(self, message):
        return _flash_s5_hooks.worker_rpc(self, "cancel", message)

    def flash_s5_transition(self, message):
        return _flash_s5_hooks.worker_rpc(self, "transition", message)

    def flash_s5_bootstrap(self, message):
        return _flash_s5_hooks.worker_rpc(self, "bootstrap", message)

'''


def _sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def _replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('source anchor changed or duplicated: ' +
                         old[:100].replace('\n', '\\n'))
    return text.replace(old, new, 1)


def _require(text, digest, path):
    got = _sha(text)
    if got != digest:
        raise ValueError(f'unreviewed source {path}: {got}')
    if '_flash_s5' in text:
        raise ValueError(f'{path} already carries E3 handoff hooks')
    return text


# --- executor ---------------------------------------------------------------

_EXECUTOR_FORWARD = (
    ('logger = init_logger(__name__)\n', 'logger = init_logger(__name__)\n\n' + _IMPORT),
    ('    def collective_rpc(  # type: ignore[override]\n',
     _EXECUTOR_WRAPPER + '    def _flash_s5_stock_collective_rpc(  # type: ignore[override]\n'),
    ('''        """Returns single result if unique_reply_rank and/or an output
        aggregator is provided, otherwise list."""
''', '''        """Returns single result if unique_reply_rank and/or an output
        aggregator is provided, otherwise list."""
''' + _EXECUTOR_GUARD),
    ('    def shutdown(self):\n        """Properly shut down the executor and its workers"""\n',
     '    def shutdown(self):\n'
     '        return _flash_s5_hooks.executor_shutdown(self, self._flash_s5_stock_shutdown)\n\n'
     '    def _flash_s5_stock_shutdown(self):\n        """Properly shut down the executor and its workers"""\n'
     '        _flash_s5_hooks.before_shutdown(self)\n'),
    ('        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):\n',
     '        if (rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None)) and not _flash_s5_hooks.retain_mq(self, None):\n'),
    ('''                for w in workers:
                    # Shutdown response queues
                    if w.worker_response_mq is not None:
                        w.worker_response_mq.shutdown()
                        w.worker_response_mq = None
''', '''                for w in workers:
                    # Shutdown response queues, but retain the MQ when the owned
                    # receiver could not prove every process was retired.
                    if w.worker_response_mq is not None and not _flash_s5_hooks.retain_mq(self, w):
                        w.worker_response_mq.shutdown()
                        w.worker_response_mq = None
'''),
    ('        if response_mqs := getattr(self, "response_mqs", None):\n',
     '        if (response_mqs := getattr(self, "response_mqs", None)) and not _flash_s5_hooks.retain_mq(self, None):\n'),
)


def generate_executor(source):
    source = _require(source, EXECUTOR_SHA256, EXECUTOR_PATH)
    text = source
    for old, new in _EXECUTOR_FORWARD:
        text = _replace(text, old, new)
    import ast
    ast.parse(text)
    return text


def revert_executor(source, expected=None):
    text = source
    for old, new in reversed(_EXECUTOR_FORWARD):
        text = _replace(text, new, old)
    if expected is not None and text != expected:
        raise ValueError('executor revert did not recover the supplied source')
    if expected is None and _sha(text) != EXECUTOR_SHA256:
        raise ValueError('executor revert did not recover the pinned source')
    return text


# --- engine core ------------------------------------------------------------

_NATIVE_CONTROL_METHOD = '''    def flash_s5_native_control(self, command):
        """Bounded named utility: one typed native-control intent or query."""
        return _flash_s5_hooks.native_control(self, command)

'''

# Native38: fatal pipes are independent of output-rank FIFO replies.
_EXECUTOR_FORWARD += (
    ('    ready_pipe: Connection\n    death_writer: Connection | None = None\n',
     '    ready_pipe: Connection\n    death_writer: Connection | None = None\n    fatal_reader: Connection | None = None\n'),
    ('    peer_worker_response_mqs: list[MessageQueue | None]\n    death_writer: Connection | None = None\n',
     '    peer_worker_response_mqs: list[MessageQueue | None]\n    death_writer: Connection | None = None\n    fatal_reader: Connection | None = None\n'),
    ('            death_writer=unready_handle.death_writer,\n',
     '            death_writer=unready_handle.death_writer,\n            fatal_reader=unready_handle.fatal_reader,\n'),
    ('        death_reader, death_writer = context.Pipe(duplex=False)\n',
     '        death_reader, death_writer = context.Pipe(duplex=False)\n        fatal_reader, fatal_writer = _flash_s5_hooks.make_fatal_pipe(context)\n'),
    ('            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))\n',
     '            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))\n            if fatal_reader is not None: inherited_fds.append(fatal_reader.fileno())\n'),
    ('            "death_pipe": death_reader,\n',
     '            "death_pipe": death_reader,\n            "fatal_writer": fatal_writer,\n'),
    ('        death_reader.close()\n',
     '        death_reader.close()\n        if fatal_writer is not None: fatal_writer.close()\n'),
    ('        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)\n',
     '        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer, fatal_reader)\n'),
    ('                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())\n',
     '                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())\n                    if unready_worker_handle.fatal_reader is not None:\n                        inherited_fds.append(unready_worker_handle.fatal_reader.fileno())\n'),
    ('            self.workers = WorkerProc.wait_for_ready(unready_workers)\n',
     '            self.workers = WorkerProc.wait_for_ready(unready_workers)\n            _flash_s5_hooks.monitor_fatal_workers(self)\n'),
    ('        death_pipe = kwargs.pop("death_pipe", None)\n',
     '        death_pipe = kwargs.pop("death_pipe", None)\n        _flash_s5_hooks.set_fatal_writer(kwargs.pop("fatal_writer", None))\n'),
    ('            logger.exception("WorkerProc hit an exception.")\n',
     '            _flash_s5_hooks.report_worker_failure(self.rank, method, e)\n            logger.exception("WorkerProc hit an exception.")\n'),
    ('            output = self.async_output_queue.get()\n            self.enqueue_output(output)\n',
     '            output = self.async_output_queue.get()\n            try:\n                self.enqueue_output(output)\n            except Exception as error:\n                _flash_s5_hooks.report_worker_failure(self.rank, "async_output", error)\n                raise\n'),
    ('            _self.shutdown()\n            callback = _self.failure_callback\n',
     '            _flash_s5_hooks.notify_worker_failure(_self, None)\n            _self.shutdown()\n            callback = _self.failure_callback\n'),
)

_CORE_FORWARD = (
    ('logger = init_logger(__name__)\n', 'logger = init_logger(__name__)\n\n' + _IMPORT),
    ('''        self.is_ec_consumer = (
''', '''        _flash_s5_hooks.after_engine_startup(self)
        self.is_ec_consumer = (
'''),
    ('    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:\n',
     _NATIVE_CONTROL_METHOD + '    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:\n'),
    ('''        batch_queue = self.batch_queue
        assert batch_queue is not None
''', '''        batch_queue = self.batch_queue
        assert batch_queue is not None
        _flash_s5_tick = _flash_s5_hooks.engine_idle_tick(self)
        if _flash_s5_tick is not None:
            return _flash_s5_tick
        _flash_s5_ready = _flash_s5_hooks.before_step_with_batch_queue(self)
        if _flash_s5_ready is not None:
            return _flash_s5_ready
'''),
    ('''        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
''', '''        _flash_s5_tick = _flash_s5_hooks.engine_idle_tick(self)
        if _flash_s5_tick is not None:
            return _flash_s5_tick
        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
'''),
    ('''            or bool(self.batch_queue)
        )
''', '''            or bool(self.batch_queue)
            or _flash_s5_hooks.native_work_pending(self)
        )
'''),
    ('''        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            with self.log_error_detail(scheduler_output):
''', '''        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            if _flash_s5_hooks.park_prefill(self, scheduler_output):
                # E3 owns this chunk: it runs as a typed asynchronous operation
                # on the workers. The engine must not launch stock work for it,
                # and its single update_from_output happens on completion.
                return None, False
            with self.log_error_detail(scheduler_output):
'''),
    ('''    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""
''', '''    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""
        _flash_s5_hooks.before_abort(self, request_ids)
'''),
    ('''        if mode == "abort":
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
''', '''        _flash_s5_hooks.before_pause(self, mode)
        if mode == "abort":
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
'''),
    ('''        if mode == "abort":
            aborted_reqs = self.scheduler.finish_requests(
''', '''        _flash_s5_hooks.before_pause(self, mode)
        if mode == "abort":
            aborted_reqs = self.scheduler.finish_requests(
'''),
    ('        self.model_executor.collective_rpc("synchronize_device")\n',
     '        _flash_s5_hooks.before_finish_pause(self)\n'
     '        self.model_executor.collective_rpc("synchronize_device")\n'),
    ('''    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
''', '''    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        _flash_s5_hooks.before_reset_prefix(self)
        return self.scheduler.reset_prefix_cache(
'''),
    ('        logger.debug_once("[shutdown] EngineCore: tearing down local resources")\n',
     '        _flash_s5_hooks.before_engine_shutdown(self)\n'
     '        logger.debug_once("[shutdown] EngineCore: tearing down local resources")\n'),
)


_CORE_FORWARD += (
    ('        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:\n            raise RuntimeError("Executor failed.")\n',
     '        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:\n            _flash_s5_hooks.raise_worker_failure(self)\n            raise RuntimeError("Executor failed.")\n'),
    ('    def _make_ready_response(self) -> EngineCoreReadyResponse:\n',
     '    def _make_ready_response(self) -> EngineCoreReadyResponse:\n        _flash_s5_hooks.require_startup_owner(self)\n'),
)


def generate_core(source):
    source = _require(source, CORE_SHA256, CORE_PATH)
    text = source
    for old, new in _CORE_FORWARD:
        text = _replace(text, old, new)
    import ast
    ast.parse(text)
    return text


def revert_core(source, expected=None):
    text = source
    for old, new in reversed(_CORE_FORWARD):
        text = _replace(text, new, old)
    if expected is not None and text != expected:
        raise ValueError('core revert did not recover the supplied source')
    if expected is None and _sha(text) != CORE_SHA256:
        raise ValueError('core revert did not recover the pinned source')
    return text


# --- worker -----------------------------------------------------------------

_WORKER_FORWARD = (
    ('logger = init_logger(__name__)\n', 'logger = init_logger(__name__)\n\n' + _IMPORT),
    ('    def shutdown(self) -> None:\n', _WORKER_METHODS + '    def shutdown(self) -> None:\n'),
)


def generate_worker(source):
    source = _require(source, WORKER_SHA256, WORKER_PATH)
    text = source
    for old, new in _WORKER_FORWARD:
        text = _replace(text, old, new)
    import ast
    ast.parse(text)
    return text


def revert_worker(source, expected=None):
    text = source
    for old, new in reversed(_WORKER_FORWARD):
        text = _replace(text, new, old)
    if expected is not None and text != expected:
        raise ValueError('worker revert did not recover the supplied source')
    if expected is None and _sha(text) != WORKER_SHA256:
        raise ValueError('worker revert did not recover the pinned source')
    return text


# --- owned executor transport ----------------------------------------------

class LocalFuture:
    """Nonblocking future facade over the owned receiver.

    ``done``/``poll`` only inspect local state. ``result`` is an explicit
    blocking path for ordinary serial callers; it polls this call only and
    never drains an unresolved earlier future.
    """

    def __init__(self, executor, sequence, aggregate=None):
        self.executor = executor
        self.sequence = sequence
        self.aggregate = aggregate
        self._lock = threading.RLock()
        self._consumed = False
        self._taken = None
        self._taken_lag = None
        self._result_ready = False
        self._result = None
        self._exception = None

    def _transport(self):
        transport = getattr(self.executor, '_flash_s5_transport', None)
        if transport is None:
            raise TransportError('owned reply transport is not installed')
        return transport

    def poll(self, budget=1):
        with self._lock:
            if self._consumed:
                return True
            transport = self._transport()
            transport.poll(budget)
            return transport.call_ready(self.sequence)

    def done(self):
        return self.poll(1)

    def take(self):
        with self._lock:
            if self._consumed:
                raise NotReady('future result already consumed')
            value, lag = self._transport().take(self.sequence)
            self._consumed = True
            self._taken, self._taken_lag = value, lag
            return value

    def result(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                if self._result_ready:
                    if self._exception is not None:
                        raise self._exception
                    return self._result
                if not self._consumed:
                    transport = self._transport()
                    transport.poll(1)
                    if transport.call_ready(self.sequence):
                        self.take()
                if self._consumed:
                    try:
                        self._result = self.aggregate(self._taken) if self.aggregate else self._taken
                    except Exception as exc:
                        self._exception = exc
                        raise
                    finally:
                        self._result_ready = True
                    return self._result
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('reply not ready before deadline')
            time.sleep(0.001)


class StockLocalFuture(LocalFuture):
    """Stock RPC lifecycle over the sole receiver, without a second MQ reader.

    Stock EngineCore intentionally resolves the sampling future and discards
    its earlier execute_model future on success. FutureWrapper.result claims
    that earlier reply too. Preserve that contract for ordinary RPCs only;
    native typed calls and LocalFuture.done/poll/take remain independent.
    """

    def __init__(self, executor, sequence, aggregate=None):
        super().__init__(executor, sequence, aggregate)
        with self._transport()._condition:
            executor._flash_s5_stock_futures[sequence] = self

    def take(self):
        with self._lock:
            value = super().take()
            with self._transport()._condition:
                self.executor._flash_s5_stock_futures.pop(self.sequence, None)
            return value

    def result(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        # Registration occurs inside submit's writer section, before enqueue.
        # Snapshot only ordinary RPCs. Never claim a native operation's ACK.
        with self._transport()._condition:
            preceding = sorted((seq, future) for seq, future in
                               self.executor._flash_s5_stock_futures.items()
                               if seq < self.sequence)
        for _seq, future in preceding:
            remaining = None if deadline is None else max(0., deadline - time.monotonic())
            try:
                LocalFuture.result(future, remaining)
            except Exception:
                # Stock FutureWrapper stores an earlier future's exception;
                # resolving a later successful future does not raise it. The
                # transport still propagates global poison on our own result.
                if not future._result_ready:
                    raise
        remaining = None if deadline is None else max(0., deadline - time.monotonic())
        return super().result(remaining)


def _reply_ranks(executor, unique_reply_rank):
    world = len(executor.response_mqs)
    if unique_reply_rank is None:
        return tuple(range(world))
    if type(unique_reply_rank) is not int or not 0 <= unique_reply_rank < world:
        raise E3ContractError('invalid unique reply rank')
    return (unique_reply_rank,)


class BoundedSender:
    """Strict bounded inline sender for the one broadcast ring.

    A command is written only when the ring already has a writable slot and the
    serialized envelope fits: no queue growth, no blocking wait and no overflow
    path (remote readers are refused outright). ``check`` runs BEFORE any reply
    expectation is registered, so a full ring is plain backpressure -- it never
    poisons the owned transport. The first write failure after registration
    quarantines the transport (partial-send), and no command is retried here.
    """

    def __init__(self, queue):
        if queue is None or not callable(getattr(queue, 'enqueue', None)):
            raise E3ContractError('bounded sender needs the broadcast queue')
        if getattr(queue, '_is_writer', True) is not True:
            raise E3ContractError('bounded sender must own the writer side')
        owner = getattr(queue, '_flash_s5_sender_owner', None)
        if owner is not None and owner is not self:
            raise E3ContractError('broadcast queue already has an E3 sender')
        queue._flash_s5_sender_owner = self
        self.queue = queue
        self.sent = 0
        self.refused_full = 0
        self.refused_large = 0

    def _writable(self):
        buffer = getattr(self.queue, 'buffer', None)
        metadata = getattr(buffer, 'get_metadata', None)
        index = getattr(self.queue, 'current_idx', None)
        if not callable(metadata) or type(index) is not int:
            raise E3ContractError('broadcast ring metadata is unavailable')
        with metadata(index) as slots:
            return not (slots[0] and sum(slots[1:]) != buffer.n_reader)

    def _check_size(self, envelope):
        import pickle
        try:
            size = len(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
        except BaseException:
            raise E3ContractError('command envelope is not serializable')
        buffer = getattr(self.queue, 'buffer', None)
        limit = getattr(buffer, 'max_chunk_bytes', None)
        if type(limit) is not int or limit < 7:
            raise E3ContractError('broadcast inline capacity is unavailable')
        if size + 6 >= limit:
            self.refused_large += 1
            raise Backpressure('command_too_large')

    def check(self, envelope):
        """Pure precheck: no registration, no write, no state change on success."""
        if getattr(self.queue, '_flash_s5_sender_owner', None) is not self:
            raise E3ContractError('bounded sender lost sole-writer ownership')
        if getattr(self.queue, 'n_remote_reader', 0):
            raise Backpressure('remote_readers_unsupported')
        if getattr(self.queue, 'shutting_down', False):
            raise TransportError('broadcast queue is shutting down')
        self._check_size(envelope)
        if not self._writable():
            self.refused_full += 1
            raise Backpressure('command_ring_full')
        return True

    def __call__(self, envelope):
        """One attempt. A failure here may be a partial send: never retry."""
        self.check(envelope)
        self.queue.enqueue(envelope)
        self.sent += 1
        return self.sent

    def __repr__(self):
        return (f'BoundedSender(sent={self.sent}, full={self.refused_full}, '
                f'large={self.refused_large})')


_INSTALL_LOCK = threading.RLock()


def install(executor, *, config, success_status, clock=time.monotonic,
            capacity=None):
    current = getattr(executor, '_flash_s5_transport', None)
    if current is not None:
        current._owner()  # Refuse inherited parent resources before any lock.
    with _INSTALL_LOCK:
        if getattr(executor, '_flash_s5_shutdown_started', False):
            raise E3ContractError('executor shutdown already started')
        return _install_owned(executor, config=config, success_status=success_status,
                              clock=clock, capacity=capacity)


def _install_owned(executor, *, config, success_status, clock=time.monotonic,
            capacity=None):
    """Take sole receiver ownership at an idle executor handoff.

    The caller must pass the actual ``WorkerProc.ResponseStatus.SUCCESS`` enum.
    Any pending stock future, an existing response reader, shared queue object
    or a non-idle executor refuses installation. Reply credits are bounded by
    the actual broadcast ring capacity, not by a generic adapter default.
    """
    if not config.enabled:
        raise E3ContractError('overlap is disabled')
    if executor is None:
        raise E3ContractError('executor is required')
    if getattr(executor, '_flash_s5_fatal_failure', None) is not None:
        raise executor._flash_s5_fatal_failure
    if success_status is None:
        raise E3ContractError('actual WorkerProc.ResponseStatus.SUCCESS is required')
    if getattr(executor, '_flash_s5_transport', None) is not None:
        raise E3ContractError('executor already has an owned receiver')
    pending = getattr(executor, 'futures_queue', None)
    if pending:
        raise E3ContractError('executor handoff is not idle: unresolved futures')
    queues = getattr(executor, 'response_mqs', None)
    if not queues:
        raise E3ContractError('executor has no response queues')
    if len({id(q) for q in queues}) != len(queues):
        raise E3ContractError('response queue object is shared between ranks')
    broadcast = getattr(executor, 'rpc_broadcast_mq', None)
    if broadcast is None:
        raise E3ContractError('executor has no broadcast queue')
    from async_handoff import derive_command_capacity
    slots = derive_command_capacity(broadcast, capacity or config.command_slots)
    transport = ReplyTransport(dict(enumerate(queues)), success_status,
                               handoff_idle=True, max_calls=slots,
                               clock=clock)
    executor._flash_s5_stock_futures = {}
    executor._flash_s5_transport = transport
    executor._flash_s5_bounded_send = BoundedSender(broadcast)
    executor._flash_s5_command_slots = slots
    executor._flash_s5_stock_forbidden = True
    executor._flash_s5_close = None
    executor._flash_s5_clock = clock
    return transport


def collective_rpc(executor, stock, method, timeout=None, args=(), kwargs=None,
                   non_block=False, unique_reply_rank=None,
                   kv_output_aggregator=None, ec_output_aggregator=None):
    """Generated ``collective_rpc`` entry point.

    With no owned transport the stock body runs unchanged (default-off
    identity). With a transport installed, every reply is registered before the
    broadcast and read by the owned receiver only.
    """
    transport = getattr(executor, '_flash_s5_transport', None)
    if transport is None:
        return stock(method, timeout=timeout, args=args, kwargs=kwargs,
                     non_block=non_block, unique_reply_rank=unique_reply_rank,
                     kv_output_aggregator=kv_output_aggregator,
                     ec_output_aggregator=ec_output_aggregator)
    if getattr(executor, '_flash_s5_stock_forbidden', False) is not True:
        raise E3ContractError('owned transport installed without disabling stock route')
    if getattr(executor, 'is_failed', False):
        raise TransportError('executor failed')
    kwargs = kwargs or {}
    if isinstance(method, str):
        send_method = method
        kind = method
    else:
        import cloudpickle
        import pickle
        send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        kind = 'callable'
    ranks = _reply_ranks(executor, unique_reply_rank)
    aggregators = [a for a in (kv_output_aggregator, ec_output_aggregator) if a]
    if aggregators:
        ranks = tuple(range(len(executor.response_mqs)))
        rank = unique_reply_rank or 0

        def aggregate(outputs):
            result = outputs[rank]
            for aggregator in aggregators:
                result = aggregator.aggregate(outputs, output_rank=rank)
            return result

        broadcast_rank = None
    else:
        def aggregate(outputs):
            return outputs[0] if unique_reply_rank is not None else outputs

        broadcast_rank = unique_reply_rank
    effective_timeout = timeout if timeout is not None else config_reply_timeout(executor)
    if type(effective_timeout) not in (int, float) or not math.isfinite(effective_timeout) or effective_timeout <= 0:
        raise E3ContractError('executor RPC needs a positive finite timeout')
    future = None
    def send(sequence):
        nonlocal future
        # Register in the same serialized section as the reply expectation and
        # broadcast, so a second stock caller cannot omit an earlier future.
        future = StockLocalFuture(executor, sequence, aggregate=aggregate)
        executor.rpc_broadcast_mq.enqueue((send_method, args, kwargs,
                                           broadcast_rank))
    sequence = transport.submit(kind, send, ranks=ranks,
                                timeout_s=effective_timeout, aggregate=None)
    return future if non_block else future.result(timeout=effective_timeout)


def config_reply_timeout(executor):
    """The configured reply budget is carried on the owner; default is finite."""
    handoff = getattr(executor, '_flash_s5_handoff', None)
    if handoff is not None:
        return handoff.config.reply_timeout_s
    return 300.0


def executor_shutdown(executor, stock):
    """Serialize the full stock teardown, not only the transport close call.

    Otherwise a second monitor/finalizer can skip the worker-termination wait
    and destroy MQs while the first shutdown still waits for live workers.
    """
    current = getattr(executor, '_flash_s5_transport', None)
    if current is not None:
        current._owner()
    with _INSTALL_LOCK:
        executor._flash_s5_shutdown_started = True
        current = getattr(executor, '_flash_s5_transport', None)
    if current is None:
        return stock()
    current._owner()
    with current._shutdown_lock:
        return stock()


def before_shutdown(executor):
    """Close the owned receiver before response MQs are destroyed.

    Returns ``(clean, report)``. An unclean close keeps the transport and its
    MQs owned; ``retain_mq`` then refuses to tear those queues down. No forced
    thread cancellation or lease retirement is implied.
    """
    current = getattr(executor, '_flash_s5_transport', None)
    if current is not None:
        current._owner()  # Forked processes cannot acquire parent lifecycle locks.
    with _INSTALL_LOCK:
        executor._flash_s5_shutdown_started = True
        transport = getattr(executor, '_flash_s5_transport', None)
    if transport is None:
        return True, None
    transport._owner()  # Same process, including IO/monitor/finalizer threads.
    with transport._shutdown_lock:
        close = getattr(executor, '_flash_s5_close', None)
        if close is not None:
            return bool(close.get('clean')), close
        # Concurrent shutdown paths must retain MQs while close is in flight.
        executor._flash_s5_close = {'clean': False, 'scope': 'transport close pending'}
        report = transport.close(2.0)
        executor._flash_s5_close = report
        return bool(report.get('clean')), report


def retain_mq(executor, worker):
    """True when the stock shutdown must keep this response MQ alive."""
    close = getattr(executor, '_flash_s5_close', None)
    if close is None:
        return False
    return not close.get('clean', False)


# --- engine-side hooks ------------------------------------------------------

def _handoff(engine):
    handoff = getattr(engine, '_flash_s5_handoff', None)
    if handoff is not None and not hasattr(handoff, 'poll'):
        raise E3ContractError('engine E3 handoff is not a HandoffDispatcher')
    return handoff


def _driver(engine):
    driver = getattr(engine, '_flash_s5_driver', None)
    if driver is not None and not callable(getattr(driver, 'advance', None)):
        raise E3ContractError('engine E3 segment driver is invalid')
    return driver


def after_engine_startup(engine):
    """Generated ``EngineCore.__init__`` seam: install the native composition."""
    from s5_startup import install_engine

    return install_engine(engine)


def park_prefill(engine, scheduler_output):
    """True when E3 took this whole scheduled step as an asynchronous chunk."""
    driver = _driver(engine)
    if driver is None:
        return False
    parked = driver.park(scheduler_output)
    return parked is not None and parked is not False


def before_step_with_batch_queue(engine):
    """Bounded engine bypass: drive one typed step; never wait for a layer."""
    driver = _driver(engine)
    if driver is not None:
        return _stock_step_bypass(driver, driver.advance())
    handoff = _handoff(engine)
    if handoff is None:
        return None
    report = handoff.poll(8)
    if report.failed:
        raise E3ContractError('E3 handoff is quarantined')
    return None


def native_control(engine, command):
    """Generated ``EngineCore.flash_s5_native_control`` seam.

    Returns the driver's ``concurrent.futures.Future`` for an intent so the
    core defers the utility reply instead of pumping worker replies inline.
    """
    if type(command) is not dict:
        raise E3ContractError('native control command must be a bounded dict')
    driver = _driver(engine)
    if driver is None:
        raise E3ContractError('native control requires an installed driver')
    submit = getattr(driver, 'submit_control', None)
    if not callable(submit):
        raise E3ContractError('native driver has no control-intent seam')
    return submit(command)


def control_pending(engine):
    """True while one bounded control intent still needs engine ticks."""
    driver = getattr(engine, '_flash_s5_driver', None)
    pending = getattr(driver, 'control_pending', None)
    return bool(pending()) if callable(pending) else False


def native_work_pending(engine):
    """Keep the process loop awake for native ownership, even while paused."""
    driver = getattr(engine, '_flash_s5_driver', None)
    return bool(getattr(driver, 'active', False)) or control_pending(engine)


def _stock_step_bypass(driver, result):
    if result is not None:
        note = getattr(driver, 'note_stock_step_skipped', None)
        if callable(note):
            note()
    return result


def engine_idle_tick(engine):
    """Advance one bounded control/paused-native tick without a reply wait.

    PAUSED_ALL hides ordinary scheduler work. Native ownership must nevertheless
    keep EngineCoreProc out of its blocking input wait and prevent an idle
    callback from finishing pause before the queried stop/transition settles.
    Normal active stock steps retain their existing driver seam and quantum.
    """
    driver = getattr(engine, '_flash_s5_driver', None)
    tick = getattr(driver, 'control_tick', None)
    pump = getattr(driver, 'pump_control', None)
    if not callable(tick) and not callable(pump):
        return None
    if control_pending(engine):
        if callable(tick):
            tick()
        elif callable(pump):
            pump()
        # One bounded control advance per engine tick; settlement does not
        # license a second native/stock submission in this same tick.
        if control_pending(engine):
            time.sleep(0.001)
        return _stock_step_bypass(driver, ({}, False))
    scheduler = getattr(engine, 'scheduler', None)
    has_requests = getattr(scheduler, 'has_requests', None)
    if (getattr(driver, 'active', False)
            and (getattr(driver, 'stop_kind', None) is not None
                 or (callable(has_requests) and not has_requests()))):
        # PAUSED_ALL hides scheduler requests. A native stop/partial transition
        # still owns its reply and must settle before the process idle callback
        # may finish pause or block waiting for a new client request.
        output = driver.advance()
        if getattr(driver, 'active', False):
            time.sleep(0.001)
        # A bounded native refusal may complete a parked request while stock
        # work is hidden. Preserve its normal EngineCore output for the client.
        return _stock_step_bypass(driver, output if output is not None else ({}, False))
    return None


def before_abort(engine, request_ids):
    driver = _driver(engine)
    if driver is not None:
        if hasattr(driver, 'tx'):
            # Native driver: abort may target a begun transaction or a chunk
            # that is only parked in the admission controller.
            ids = tuple(request_ids)
            transaction = driver.tx
            if transaction is None and driver.controller.active:
                transaction = driver.controller.admission().transaction
            if transaction is not None and transaction.request_uid in ids:
                return driver.stop('abort', drain=False)
            retained = getattr(driver, 'retained_request', None)
            retained = retained() if callable(retained) else None
            if retained is not None and retained.request_uid in ids:
                # Between chunks the request row is owned with no chunk and no
                # parked admission; abort must close it through the same typed
                # worker cancellation, never leave it stranded.
                return driver.stop('abort', drain=False)
            return None
        active = driver.dispatcher.active
        if active is not None and active.transaction.request_uid in tuple(request_ids):
            return driver.stop('abort', drain=False)
        return None
    handoff = _handoff(engine)
    if handoff is None:
        return None
    active = handoff.active
    if active is not None and active.transaction.request_uid in tuple(request_ids):
        return handoff.stop('abort', drain=False)
    return None


def before_pause(engine, mode):
    driver = _driver(engine)
    if driver is not None:
        return driver.stop('pause', drain=False)
    handoff = _handoff(engine)
    if handoff is None:
        return None
    return handoff.stop('pause', drain=False)


def before_finish_pause(engine):
    driver = _driver(engine)
    if driver is not None:
        driver.finish_pause()
        return True
    handoff = _handoff(engine)
    if handoff is None:
        return None
    if handoff.active is not None and handoff.active.quarantine is not None:
        raise E3ContractError('pause cannot complete with an unclean transaction')
    report = handoff.poll(8)
    if handoff.active is not None:
        raise E3ContractError('pause requires asynchronous drain before device synchronization')
    return report


def before_reset_prefix(engine):
    driver = _driver(engine)
    if driver is not None:
        driver.stop('reset', drain=False)
        if driver.active:
            raise E3Pending('reset waits for parked prefill retirement')
        return None
    handoff = _handoff(engine)
    if handoff is None:
        return None
    if handoff.active is not None:
        # A parked transaction still owns unpublished source blocks.
        return handoff.stop('reset', drain=False)
    return None


def before_engine_shutdown(engine):
    driver = _driver(engine)
    if driver is not None:
        driver.stop('shutdown', drain=False)
        if driver.active:
            raise E3Pending('shutdown waits for parked prefill retirement')
        return None
    handoff = _handoff(engine)
    if handoff is None:
        return None
    return handoff.stop('shutdown', drain=False)


# --- worker-side hook -------------------------------------------------------

def _wire_method(message):
    """Exact wire-v1 record -> generated method name; ``None`` for legacy RPCs."""
    from wire_v1 import (BootstrapCommand, CancelRequest, OpenRequest,
                         PPReceiptDelivery, PeerBinding, PollRequest, QsaProfileQuery,
                         ReleaseRequest, SegmentCommand, TransitionCommand)
    for record, name in ((BootstrapCommand, 'bootstrap'),
                         (OpenRequest, 'open'), (SegmentCommand, 'submit'),
                         (PollRequest, 'poll'), (ReleaseRequest, 'release'),
                         (QsaProfileQuery, 'poll'),
                         (CancelRequest, 'cancel'), (PeerBinding, 'bind'),
                         (PPReceiptDelivery, 'receipt'),
                         (TransitionCommand, 'transition')):
        if type(message) is record:
            return name
    return None


def _wire_bridge(worker):
    for owner in (worker, getattr(worker, 'model_runner', None)):
        if owner is None:
            continue
        bridge = getattr(owner, '_flash_s5_bridge', None)
        if bridge is not None:
            return bridge
    raise E3ContractError('worker has no native LaneDispatcher bridge')


def worker_rpc(worker, name, message):
    """Generated worker entry point for typed open/submit/poll/commit/cancel."""
    wire_name = _wire_method(message)
    if wire_name is not None:
        # Wire-v1 records carry their own exact type; they have no legacy
        # ``kind``/``validate`` adapter and never fall through to it.
        if name != wire_name:
            raise E3ContractError('S5 RPC method and wire record disagree')
        bridge = _wire_bridge(worker)
        attach = getattr(bridge, 'attach', None)
        if callable(attach):
            attach(worker)
        if name in ('bind', 'receipt') and message.sender_rank != bridge.rank:
            # The stock MQ executes every broadcast on every worker; selecting
            # output_rank only selects the reply. Non-senders must do no work
            # and return quietly instead of logging a spurious worker error.
            return None
        handler = getattr(bridge, wire_name, None)
        if not callable(handler):
            raise E3ContractError('native bridge lacks the requested S5 RPC')
        return handler(message)
    if name in ('bind', 'receipt', 'release', 'bootstrap'):
        raise E3ContractError('S5 RPC requires its exact wire record')
    if name not in ('open', 'submit', 'poll', 'commit', 'cancel', 'transition',
                    'bootstrap'):
        raise E3ContractError('unknown S5 worker RPC')
    if not hasattr(message, 'validate'):
        raise E3ContractError('S5 worker RPC requires a typed message')
    message.validate()
    if getattr(message, 'kind', None) != name:
        raise E3ContractError('S5 RPC method and typed message disagree')
    adapter = getattr(worker, '_flash_s5_worker', None)
    if adapter is None:
        runner = getattr(worker, 'model_runner', None)
        adapter = getattr(runner, '_flash_s5_worker', None) if runner is not None else None
    if adapter is None:
        raise E3ContractError('worker has no C S5 worker adapter')
    attach = getattr(adapter, 'attach_worker', None)
    if callable(attach):
        # The generated GPU worker is the object that launches the forward.
        attach(worker)
    handler = getattr(adapter, name, None)
    if handler is None:
        raise E3ContractError('worker adapter lacks the requested S5 RPC')
    return handler(message)


__all__ = [
    'BoundedSender', 'CORE_PATH', 'CORE_SHA256', 'EXECUTOR_PATH',
    'EXECUTOR_SHA256', 'LocalFuture', 'PINS', 'WORKER_PATH', 'WORKER_SHA256',
    'after_engine_startup', 'before_abort', 'before_engine_shutdown',
    'before_finish_pause', 'before_pause', 'before_reset_prefix',
    'before_shutdown', 'before_step_with_batch_queue', 'collective_rpc',
    'config_reply_timeout', 'control_pending', 'engine_idle_tick', 'executor_shutdown',
    'generate_core', 'generate_executor',
    'generate_worker', 'install', 'log_worker_failure', 'native_control', 'native_work_pending', 'park_prefill',
    'retain_mq', 'revert_core',
    'revert_executor', 'revert_worker', 'worker_rpc',
]


_FATAL_WRITER = None
_FATAL_LOCK = threading.Lock()
_FATAL_SENT = False
FATAL_RECORD_MAX = 1024  # monitor_fatal_workers reads one record with recv_bytes(1024)
# Native43 le-fix (b): bounded metadata that lets the parent log the cause.  ``where`` and ``cause`` are code
# locations and type names; ``detail`` is the message of this package's own typed errors only (foreign exception
# payloads, e.g. torch or vLLM messages, are never serialized: WorkerRPCError's contract).
FATAL_OPTIONAL = {'detail': 160, 'where': 100, 'cause': 80}


def _fatal_own_error(error):
    module = sys.modules.get(type(error).__module__)
    path = getattr(module, '__file__', None)
    return bool(path) and os.path.dirname(os.path.abspath(path)) == os.path.dirname(os.path.abspath(__file__))


def _fatal_metadata(error):
    """CPU-only cause metadata for the fatal record; never raises."""
    rows = {}
    try:
        tb, last = error.__traceback__, None
        while tb is not None:
            last, tb = tb, tb.tb_next
        if last is not None:
            code = last.tb_frame.f_code
            rows['where'] = '%s:%d:%s' % (os.path.basename(code.co_filename), last.tb_lineno, code.co_name)
        linked = error.__cause__ if error.__cause__ is not None else error.__context__
        if linked is not None:
            rows['cause'] = type(linked).__name__
        if _fatal_own_error(error):
            rows['detail'] = str(error)
    except Exception:  # noqa: BLE001 - metadata is best effort; the record itself must go out
        pass
    return {k: v.encode('ascii', 'backslashreplace').decode('ascii')[:FATAL_OPTIONAL[k]]
            for k, v in rows.items() if type(v) is str}


def make_fatal_pipe(context):
    if os.environ.get('FLASH_S5_OVERLAP') != '1':
        return None, None
    return context.Pipe(duplex=False)


def set_fatal_writer(writer):
    global _FATAL_WRITER, _FATAL_SENT
    _FATAL_WRITER, _FATAL_SENT = writer, False


def report_worker_failure(rank, method, error):
    """First fatal notification precedes logging and normal FIFO output.

    A notified worker parks until its supervisor retires the generation. It
    cannot continue or report success after a partly executed PP forward.
    """
    global _FATAL_SENT
    if _FATAL_WRITER is None:
        return
    with _FATAL_LOCK:
        if not _FATAL_SENT:
            _FATAL_SENT = True
            record = dict(schema='s5-worker-fatal/1', rank=rank, pid=os.getpid(),
                          method=method[:80] if type(method) is str else 'callable',
                          error_type=type(error).__name__[:80], code='worker-rpc-exception')
            record.update(_fatal_metadata(error))
            data = json.dumps(record).encode()
            for key in ('detail', 'where', 'cause'):
                if len(data) <= FATAL_RECORD_MAX:
                    break
                record.pop(key, None)
                data = json.dumps(record).encode()
            try:
                _FATAL_WRITER.send_bytes(data)
            except (OSError, EOFError):
                raise SystemExit(1) from None
    # No queue/log/device work after publication. Parent death monitoring or
    # supervisor termination also retires this worker if the reader dies.
    threading.Event().wait()
    raise SystemExit(1)


def log_worker_failure(worker, record, unreadable=None):
    """Native43 le-fix (b): the parent names the shutdown cause before the teardown it starts.

    l-e (1 Nov 2026) shut the executor down from this channel and no process logged why: the worker parks
    without logging (report_worker_failure) and the parent stored the record silently.  One ERROR line,
    after the futures are poisoned and Core is notified, before executor.shutdown(); it never raises.
    """
    try:
        logger = logging.getLogger(__name__)
        if record is None:
            logger.error('FLASH_S5_WORKER_FAILURE rank=%s pid=%s record=unavailable (%s); '
                         'executor shutdown follows (worker fatal channel)',
                         getattr(worker, 'rank', '?'), getattr(getattr(worker, 'proc', None), 'pid', '?'),
                         unreadable or 'unknown')
        else:
            logger.error('FLASH_S5_WORKER_FAILURE rank=%s pid=%s method=%s error_type=%s code=%s cause=%s '
                         'where=%s detail=%r; executor shutdown follows (worker fatal channel)',
                         record.get('rank'), record.get('pid'), record.get('method'), record.get('error_type'),
                         record.get('code'), record.get('cause'), record.get('where'), record.get('detail'))
    except Exception:  # noqa: BLE001 - logging must never stop the teardown
        pass


def notify_worker_failure(executor, record):
    """Poison futures and notify Core BEFORE potentially blocking teardown."""
    lock = getattr(executor, '_flash_s5_fatal_lock', None)
    if lock is None:
        # The default-off worker monitor is left to its stock callback path.
        return
    with lock:
        if getattr(executor, '_flash_s5_fatal_failure', None) is not None:
            return
        if record is None:
            record = dict(schema='s5-worker-fatal/1', rank=-1, pid=0,
                          method='unknown', error_type='WorkerDied', code='worker-died')
        failure = WorkerRPCError(record)
        executor._flash_s5_fatal_failure = failure
        executor.is_failed = True
        transport = getattr(executor, '_flash_s5_transport', None)
        if transport is not None:
            transport.fail(failure)
        callback = getattr(executor, 'failure_callback', None)
        executor.failure_callback = None
    if callback is not None:
        callback()


def monitor_fatal_workers(executor):
    if os.environ.get('FLASH_S5_OVERLAP') != '1':
        return
    if hasattr(executor, '_flash_s5_fatal_lock'):
        raise E3ContractError('worker fatal monitor already installed')
    workers = tuple(executor.workers)
    if len(workers) != getattr(executor, 'world_size', len(workers)):
        raise E3ContractError('native fatal channels require all ranks local')
    if any(getattr(w, 'fatal_reader', None) is None for w in workers):
        raise E3ContractError('enabled executor requires every worker fatal channel')
    executor._flash_s5_fatal_lock = threading.Lock()
    executor._flash_s5_fatal_failure = None
    def receive(worker):
        unreadable = None
        try:
            raw = worker.fatal_reader.recv_bytes(FATAL_RECORD_MAX)
            record = json.loads(raw)
            if (type(record) is not dict or record.get('schema') != 's5-worker-fatal/1'
                    or record.get('rank') != worker.rank
                    or type(record.get('pid')) is not int
                    or record['pid'] != worker.proc.pid
                    or any(type(record.get(k)) is not str or len(record[k]) > 80
                           for k in ('method', 'error_type', 'code'))
                    or any(k in record and (type(record[k]) is not str or len(record[k]) > n)
                           for k, n in FATAL_OPTIONAL.items())):
                raise ValueError('invalid worker fatal record')
        except (OSError, EOFError, ValueError) as exc:
            record, unreadable = None, type(exc).__name__
        if getattr(executor, 'shutting_down', False):
            return
        try:
            notify_worker_failure(executor, record)
        finally:
            log_worker_failure(worker, record, unreadable)
            worker.fatal_reader.close()
            # Death pipes/signals, never an RPC to peers blocked in PP/NCCL.
            executor.shutdown()
    for worker in workers:
        threading.Thread(target=receive, args=(worker,), daemon=True,
                         name='flash-s5-fatal-%d' % worker.rank).start()


def raise_worker_failure(engine):
    """Keep the first typed cause when the idle Core consumes its failure notice."""
    failure = getattr(engine.model_executor, '_flash_s5_fatal_failure', None)
    if failure is not None:
        raise failure


def require_startup_owner(engine):
    """Ready publication must follow the completed engine-owned FULL fence."""
    if os.environ.get('FLASH_S5_OVERLAP') != '1':
        return
    from decode_probe import mode_settings
    if mode_settings(os.environ) is not None:
        return
    driver = getattr(engine, '_flash_s5_driver', None)
    proof = getattr(engine, '_flash_s5_startup_ready', None)
    if (driver is None or type(proof) is not dict or not driver._bootstrapped
            or proof.get('engine_epoch') != driver.engine_epoch
            or driver._split_ready or driver.admission_intent() is not None):
        raise E3ContractError('engine owner is not ready for publication')
