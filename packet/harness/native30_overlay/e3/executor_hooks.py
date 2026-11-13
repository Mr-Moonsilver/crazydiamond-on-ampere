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
from reply_transport import (Backpressure, DirectReply, NotReady, ReplyTransport, TransportError,
                             WorkerRPCError)
from stage_wire import STAGE_SKIP, StageScoped, stage_rpc
from native_measurement import (INWAIT_ENV, PROGRESS_DIRECT_INWAIT_METHODS, PROGRESS_DIRECT_METHODS, PROGRESS_META_KEY, ProgressTrace, decode_progress_meta,
                                emit_receipt_lines,
                                encode_progress_meta, progress_mode)


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
    # Native43 le-fix: the stock broadcast write (timeout=None) through the bounded hook; with the
    # fatal channel absent (overlap off) the hook makes the identical unbounded call.
    ('        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))\n',
     '        _flash_s5_hooks.bounded_enqueue(self, self.rpc_broadcast_mq,\n'
     '                                        (send_method, args, kwargs, output_rank))\n'),
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

# Held-f: the worker reply seam.  With no progress meta on the envelope and
# FLASH_S5_NATIVE_PROGRESS unset, every hook takes exactly the stock branch
# (``handle_output`` and the one response-MQ write, now under a per-worker lock).
_EXECUTOR_FORWARD += (
    ('        method, args, kwargs, output_rank = rpc_request\n        try:\n',
     '        method, args, kwargs, output_rank = rpc_request\n'
     '        _flash_s5_rpc = None\n'
     '        try:\n'
     '            _flash_s5_rpc = _flash_s5_hooks.worker_rpc_begin(self, method, kwargs)\n'),
    ('            output = func(*args, **kwargs)\n\n'
     '            if output_rank is None or self.rank == output_rank:\n'
     '                self.handle_output(output)\n',
     '            output = func(*args, **kwargs)\n'
     '            _flash_s5_hooks.worker_rpc_returned(self, _flash_s5_rpc, output_rank)\n\n'
     '            if output_rank is None or self.rank == output_rank:\n'
     '                _flash_s5_hooks.worker_handle_output(self, _flash_s5_rpc, output)\n'),
    ('            if output_rank is None or self.rank == output_rank:\n'
     '                self.handle_output(e)\n',
     '            if output_rank is None or self.rank == output_rank:\n'
     '                _flash_s5_hooks.worker_handle_output(self, _flash_s5_rpc, e)\n'),
    ('        if (response_mq := self.worker_response_mq) is not None:\n'
     '            response_mq.enqueue(result)\n',
     '        _flash_s5_hooks.worker_enqueue_reply(self, result)\n'),
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
    ('        self.scheduler.add_request(request)\n',
     '        self.scheduler.add_request(request)\n'
     '        _flash_s5_hooks.engine_request_admitted(self, request)\n'),
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

# stage1-step L1 (FLASH_S5_LAZY_RECV, default off, per RPC by the engine's mark on the scheduler output): the PP receive
# runs at the runner's first read of ``.tensors``, after the CPU prep, instead of before the runner starts.  Same RPC, same
# order against stage 0's sends; only the host position inside this execute_model moves.  Unmarked, the stock branch runs.
_LAZY_RECV_CLASS = '''class FlashS5LazyIntermediateTensors(AsyncIntermediateTensors):
    """stage1-step L1: AsyncIntermediateTensors whose receive itself waits for the first ``.tensors``."""

    def __init__(self, receive) -> None:
        super().__init__({})
        self._flash_s5_receive = receive

    def wait_for_comm(self) -> None:
        receive = object.__getattribute__(self, "_flash_s5_receive")
        if receive is not None:
            self._flash_s5_receive = None
            tensor_dict, comm_handles, comm_postprocess = receive()
            assert tensor_dict is not None
            self.tensors = tensor_dict
            self._comm_handles = comm_handles
            self._comm_postprocess = comm_postprocess
        AsyncIntermediateTensors.wait_for_comm(self)

    def __getitem__(self, key):
        # The stock slice calls ``self.__class__(tensors)``; here a slice is a plain, received view.
        if isinstance(key, slice):
            return AsyncIntermediateTensors({k: v[key] for k, v in self.tensors.items()})
        return self.tensors[key]

    def flash_s5_require_received(self) -> None:
        # Fail closed: an unconsumed receive would hand stage 0's next batch to the next step.
        if object.__getattribute__(self, "_flash_s5_receive") is not None:
            raise RuntimeError("stage1-step lazy PP receive was never consumed by the runner")


'''

_WORKER_FORWARD = (
    ('logger = init_logger(__name__)\n', 'logger = init_logger(__name__)\n\n' + _IMPORT),
    ('class Worker(WorkerBase):\n', _LAZY_RECV_CLASS + 'class Worker(WorkerBase):\n'),
    ('''        if forward_pass and not get_pp_group().is_first_rank:
            tensor_dict, comm_handles, comm_postprocess = (
''', '''        if (
            forward_pass
            and not get_pp_group().is_first_rank
            and _flash_s5_hooks.lazy_recv_marked(scheduler_output)
        ):
            intermediate_tensors = FlashS5LazyIntermediateTensors(
                lambda: get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
            )
        elif forward_pass and not get_pp_group().is_first_rank:
            tensor_dict, comm_handles, comm_postprocess = (
'''),
    ('''                scheduler_output, intermediate_tensors
            )
            if (
                self.use_v2_model_runner
''', '''                scheduler_output, intermediate_tensors
            )
            if type(intermediate_tensors) is FlashS5LazyIntermediateTensors:
                intermediate_tensors.flash_s5_require_received()
            if (
                self.use_v2_model_runner
'''),
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
        # Held-f receipts: (trace, kind, submit ns) while an auto-owned native
        # chunk records, else None. Never read by the RPC lifecycle itself.
        self._progress = None
        with self._transport()._condition:
            executor._flash_s5_stock_futures[sequence] = self

    def take(self):
        with self._lock:
            progress = self._progress
            registered = None
            if progress is not None:
                call = self._transport()._calls.get(self.sequence)
                registered = None if call is None else call.registered_s
            value = super().take()
            with self._transport()._condition:
                self.executor._flash_s5_stock_futures.pop(self.sequence, None)
            if progress is not None:
                lags = [lag for lag in (self._taken_lag or {}).values()
                        if type(lag) in (int, float)]
                received = (None if registered is None or not lags
                            else int((registered + max(lags)) * 1e9))
                progress[0].trace.record('s', self.sequence, progress[1], progress[2],
                                         received, time.monotonic_ns())
            return value

    def result(self, timeout=None):
        progress = self._progress
        if progress is None:
            return self._stock_result(timeout)
        started = time.monotonic_ns()
        sends = capped = 0
        try:
            if progress[1] == 'sample_tokens' and progress[0].wait_allowed():
                transport = self._transport()

                def ready():
                    with self._lock:
                        if self._consumed or self._result_ready:
                            return True
                    transport.poll(1)
                    return transport.call_ready(self.sequence)
                deadline = None if timeout is None else time.monotonic() + timeout
                sends, capped = native_progress_wait(progress[0], ready, deadline)
                timeout = None if deadline is None else max(0., deadline - time.monotonic())
            return self._stock_result(timeout)
        finally:
            progress[0].counters['waits'] += 1
            ended = time.monotonic_ns()
            progress[0].trace.record('w', self.sequence, progress[1], started,
                                     ended, sends, capped)
            progress[0].steps.waited(started, ended)

    def _stock_result(self, timeout=None):
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


def native_progress_wait(progress, ready, deadline):
    """Held-f burst: bounded native progress while the owner waits on a sample.

    ``ready()`` reports the stock result; until it is ready, each round asks
    the driver for one permitted native advance (``progress_in_wait``) and
    sleeps the stock 1 ms when nothing was sent. Ends on ready, the deadline,
    a refused permit or ``WAIT_SEND_LIMIT`` sends (counted as capped). The
    stock resolution path then runs unchanged. Returns ``(sends, capped)``.
    """
    driver = progress.driver
    sends = capped = 0
    try:
        while not ready():
            if deadline is not None and time.monotonic() >= deadline:
                break
            if sends >= progress.WAIT_SEND_LIMIT:
                capped = 1
                break
            sent = driver.progress_in_wait()
            if sent is None:
                break
            sends += sent
            if not sent:
                time.sleep(0.001)
    finally:
        progress.counters['wait_sends'] += sends
        progress.counters['wait_capped'] += capped
    return sends, capped


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
            capacity=None, direct_replies=False):
    current = getattr(executor, '_flash_s5_transport', None)
    if current is not None:
        current._owner()  # Refuse inherited parent resources before any lock.
    with _INSTALL_LOCK:
        if getattr(executor, '_flash_s5_shutdown_started', False):
            raise E3ContractError('executor shutdown already started')
        return _install_owned(executor, config=config, success_status=success_status,
                              clock=clock, capacity=capacity, direct_replies=direct_replies)


def _install_owned(executor, *, config, success_status, clock=time.monotonic,
            capacity=None, direct_replies=False):
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
    extra = {'direct_replies': True} if direct_replies else {}
    transport = ReplyTransport(dict(enumerate(queues)), success_status,
                               handoff_idle=True, max_calls=slots,
                               clock=clock, **extra)
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
    progress = getattr(executor, '_flash_s5_progress_recording', None)
    def send(sequence):
        nonlocal future
        # Register in the same serialized section as the reply expectation and
        # broadcast, so a second stock caller cannot omit an earlier future.
        future = StockLocalFuture(executor, sequence, aggregate=aggregate)
        wire_kwargs = kwargs
        if progress is not None:
            # Held-f receipts only: the worker pops this before dispatch.
            wire_kwargs = dict(kwargs)
            wire_kwargs[PROGRESS_META_KEY] = encode_progress_meta(sequence, False)
            future._progress = (progress, kind, time.monotonic_ns())
            progress.counters['stock_rpcs'] += 1
            # Ring position for in-wait progress: native commands may follow a
            # sample_tokens, never sit between an execute_model and its sample.
            progress.last_stock_kind = kind
        if _DECODE_STEP:
            # decode-step (FLASH_S5_DECODE_SAMPLE_FAST / FLASH_S5_DECODE_ANATOMY): the per-arm tag; popped by the worker.
            side = executor.__dict__.get('_flash_s5_decode_step')
            tag = side.wire(kind, sequence) if side is not None else None
            if tag is not None:
                wire_kwargs = dict(wire_kwargs)
                wire_kwargs[_DECODE_STEP.WIRE_KEY] = tag
        bounded_enqueue(executor, executor.rpc_broadcast_mq,
                        (send_method, args, wire_kwargs, broadcast_rank))
    sequence = transport.submit(kind, send, ranks=ranks,
                                timeout_s=effective_timeout, aggregate=None)
    return future if non_block else future.result(timeout=effective_timeout)


# Native43 le-fix: one bounded slice of the broadcast write.  acquire_write raises TimeoutError
# before it claims a block, so a timed-out slice has written nothing and the next slice is the
# same single write.
ENQUEUE_SLICE_S = 1.0
LONG_WAIT_FALLBACK = (60, 'No available shared memory broadcast block found in %d seconds. This '
                          'typically happens when some processes are hanging or doing some '
                          'time-consuming work (e.g. compilation, weight/kv cache quantization).')


def _long_wait(queue):
    """The stock long-wait warning of the queue's own module (vLLM's shm_broadcast): interval, text, logger.

    Read from the already-imported module of the queue's class; no import happens here.  A queue from
    another module (CPU fixtures) gets the same text at the stock 60 s on this module's logger."""
    module = sys.modules.get(getattr(type(queue), '__module__', None))
    interval = getattr(module, 'VLLM_RINGBUFFER_WARNING_INTERVAL', None)
    message = getattr(module, 'LONG_WAIT_TIME_LOG_MSG', None)
    log = getattr(getattr(module, 'logger', None), 'info', None)
    if type(interval) in (int, float) and interval > 0 and type(message) is str and callable(log):
        return interval, message, log
    return LONG_WAIT_FALLBACK[0], LONG_WAIT_FALLBACK[1], logging.getLogger(__name__).info


def _enqueue_abandoned(executor, queue):
    """The typed reason a blocked broadcast write must stop, or None while the executor lives."""
    failure = getattr(executor, '_flash_s5_fatal_failure', None)
    if failure is not None:
        return failure  # the stored WorkerRPCError (fatal channel or worker death)
    if getattr(executor, 'is_failed', False):
        return RuntimeError('Executor failed.')
    if getattr(executor, 'shutting_down', False) or getattr(queue, 'shutting_down', False):
        return TransportError('broadcast write abandoned: executor shutting down')
    return None


def bounded_enqueue(executor, queue, envelope, *, slice_s=ENQUEUE_SLICE_S, clock=time.monotonic,
                    long_wait=None):
    """Native43 le-fix: a broadcast write cannot outlive a failed executor.

    l-e (1 Nov 2026): the EngineCore busy loop sat in ``acquire_write`` (``enqueue`` with
    ``timeout=None``) while the fatal channel marked the executor failed and shut it down, so
    ``EXECUTOR_FAILED`` was never processed and no ``ENGINE_CORE_DEAD`` was sent.  With the fatal
    channel installed (overlap on), the write waits in slices of ``slice_s``; after a timed-out
    slice a failed executor raises its stored typed failure, a shutting-down executor or queue
    raises, and a live but full ring keeps waiting with the stock long-wait warning.  The busy
    loop then dies loudly ("EngineCore encountered a fatal error") and sends ENGINE_CORE_DEAD.
    Without the fatal channel (overlap off) this is the stock unbounded call, unchanged.
    """
    if getattr(executor, '_flash_s5_fatal_lock', None) is None:
        return queue.enqueue(envelope)
    started = clock()
    warnings = 1
    while True:
        try:
            return queue.enqueue(envelope, timeout=slice_s)
        except TimeoutError:
            pass
        reason = _enqueue_abandoned(executor, queue)
        if reason is not None:
            raise reason
        interval, message, log = long_wait if long_wait is not None else _long_wait(queue)
        if clock() - started > interval * warnings:
            log(message, interval)
            warnings += 1


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


def lazy_recv_marked(scheduler_output):
    """stage1-step L1, worker: the engine's per-RPC mark; both TP peers read the same broadcast object."""
    return getattr(scheduler_output, 'flash_s5_lazy_recv', False) is True


def park_prefill(engine, scheduler_output):
    """True when E3 took this whole scheduled step as an asynchronous chunk."""
    driver = _driver(engine)
    if driver is None:
        return False
    if getattr(driver, 'lazy_recv', False) is True:
        driver.mark_lazy_recv(scheduler_output)
    steps = _steps(engine)
    if steps is not None:
        # Held-g R5: scheduler.schedule returned (an empty batch or tokens).
        steps.scheduled(scheduler_output)
    parked = driver.park(scheduler_output)
    return parked is not None and parked is not False


def before_step_with_batch_queue(engine):
    """Bounded engine bypass: drive one typed step; never wait for a layer."""
    driver = _driver(engine)
    if driver is not None:
        if (side := _decode_step_engine(engine)) is not None:
            side.step(engine)      # decode-step: the per-arm gate and the clean anatomy window (installed only)
        started = time.monotonic_ns()
        result = _stock_step_bypass(driver, driver.advance())
        sync = getattr(driver, 'sync_decode_merge', None)
        if sync is not None:
            # stage1-step L2: read by this step's schedule() (scheduler_hooks.decode_merge_cap).
            sync(engine.scheduler)
        steps = _steps(engine)
        if steps is not None:
            # Held-g R5: read after the advance, which may have begun the chunk.
            steps.advanced(started, result is not None)
            if result is None and driver.idle_wait_permit() and _stock_step_idle(engine):
                # Held-g L3 (arm idle=wait): this step's schedule() and
                # execute_model would carry nothing; wait for the owed native
                # reply instead (bounded), then return as the idle tick does.
                driver.idle_wait()
                steps.bypass('idle')
                result = _stock_step_bypass(driver, ({}, False))
        return result
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
    policy = getattr(driver, 'auto_arm', None)
    return (bool(getattr(policy, 'active', False))
            or bool(getattr(driver, 'active', False)) or control_pending(engine))


def _stock_step_bypass(driver, result):
    if result is not None:
        note = getattr(driver, 'note_stock_step_skipped', None)
        if callable(note):
            note()
    return result


def engine_idle_tick(engine):
    """The first seam of every engine step; held-g R5 starts a step here.

    With no auto-owned native chunk recording (always, with the progress
    setting unset) this is exactly ``_idle_tick``.
    """
    steps = _steps(engine)
    if steps is None:
        return _idle_tick(engine)
    steps.begin()
    result = _idle_tick(engine)
    if result is not None:
        steps.bypass('tick')
    return result


def _stock_step_idle(engine):
    """Held-g L3, engine side: a stock step now would schedule and send nothing.

    The scheduler holds no running, waiting or skipped request (a parked
    native request sits outside them) and nothing its next SchedulerOutput
    would hand the workers: no finished or preempted request id, new block to
    zero, pending CoW copy or freed encoder entry, and no KV/EC connector. No
    abort is pending. A stock future still queued must be an empty batch: its
    SchedulerOutput already reached the workers and only its empty update is
    left, which the next stock step performs as before (with two queue slots
    the stock loop always keeps one such future). Read-only peeks; an unknown
    shape answers False, so the stock step runs.
    """
    try:
        scheduler = engine.scheduler
        if any(entry[1].total_num_scheduled_tokens != 0 for entry in engine.batch_queue):
            return False
        if (scheduler.running or len(scheduler.waiting)
                or len(scheduler.skipped_waiting) or scheduler.finished_req_ids
                or scheduler.reset_preempted_req_ids or scheduler.connector is not None
                or getattr(scheduler, 'ec_connector', None) is not None):
            return False
        aborts = getattr(engine, 'aborts_queue', None)
        if aborts is not None and not aborts.empty():
            return False
        for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
            if getattr(manager, 'new_block_ids', None) or getattr(manager, '_pending_cow_copies', None):
                return False
        cache = scheduler.encoder_cache_manager
        return not any(getattr(cache, name, None) for name in ('freed', 'to_free', 'allocated'))
    except (AttributeError, TypeError, IndexError):
        return False


def _steps(engine):
    """Held-g R5: the recording chunk's step anatomy, else None."""
    progress = getattr(getattr(engine, '_flash_s5_driver', None), 'progress', None)
    if progress is None or not progress.recording:
        return None
    return progress.steps


def _idle_tick(engine):
    """Advance one bounded control/paused-native tick without a reply wait.

    PAUSED_ALL hides ordinary scheduler work. Native ownership must nevertheless
    keep EngineCoreProc out of its blocking input wait and prevent an idle
    callback from finishing pause before the queried stop/transition settles.
    Normal active stock steps retain their existing driver seam and quantum.
    """
    driver = getattr(engine, '_flash_s5_driver', None)
    policy = getattr(driver, 'auto_arm', None)
    if policy is not None and policy.tick():
        return _stock_step_bypass(driver, ({}, False))
    if (policy is not None and getattr(policy, 'control_inwait', None) is not None
            and policy.phase == 'control_observe'):
        # Only the policy's read-only identity may share ordinary decode RPC order.
        return None
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
    policy = getattr(getattr(engine, '_flash_s5_driver', None), 'auto_arm', None)
    if policy is not None:
        if policy.active:
            raise E3ContractError('pause-refused:auto-arm-cycle-owned')
        policy.manual_owned = True
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
    # host-gap batch registration (FLASH_S5_SMALL_PIECE_BATCH): sent only with the setting on.
    from small_piece_batch import SegmentBatch
    if type(message) is SegmentBatch:
        return 'submit'
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
    if type(message) is StageScoped:
        # FLASH_S5_STAGE_LOCAL only: the engine never sends this record otherwise.
        if _wire_method(message.message) != name:
            raise E3ContractError('S5 RPC method and wire record disagree')
        return stage_rpc(_wire_bridge(worker), name, message,
                         lambda inner: worker_rpc(worker, name, inner))
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


# --- held-f worker reply seam ------------------------------------------------
# Runs in WorkerProc's RPC loop and async output thread.  Default (no envelope
# meta, progress unset): ``worker_handle_output`` is ``handle_output`` and
# ``worker_enqueue_reply`` is the stock single response-MQ write under a lock.

_WORKER_PROGRESS = None
_WORKER_PROGRESS_LOCK = threading.Lock()
_REPLY_LOCKS_LOCK = threading.Lock()
_progress_log = None
# Stage-1 handler anatomy (FLASH_S5_HANDLER_ANATOMY=1, default off): None until
# first read, False when off, else this process's handler_anatomy.Recorder. Read
# only under a live progress state, so with progress unset it is never read.
_ANATOMY = None
# decode-step (FLASH_S5_DECODE_SAMPLE_FAST=1 or FLASH_S5_DECODE_ANATOMY=1, default off): None until first read, False
# when both are off, else the decode_step module. Both unset: the module is never imported and no RPC is tagged.
_DECODE_STEP = None
_DECODE_STEP_WORKER = None


class _CleanRpc(list):
    """decode-step clean-window RPC state: handler-anatomy rows only, never a native progress trace record."""


def _decode_step():
    global _DECODE_STEP
    if _DECODE_STEP is None:
        with _WORKER_PROGRESS_LOCK:
            if _DECODE_STEP is None:
                if (os.environ.get('FLASH_S5_DECODE_SAMPLE_FAST', '0') == '1'
                        or os.environ.get('FLASH_S5_DECODE_ANATOMY', '0') == '1'):
                    import decode_step
                    _DECODE_STEP = decode_step
                else:
                    _DECODE_STEP = False
    return _DECODE_STEP or None


def _decode_step_engine(engine):
    """The engine core's decode-step side on its executor, or None (switches off)."""
    module = _decode_step()
    if module is None:
        return None
    executor = getattr(engine, 'model_executor', None)
    if executor is None:
        return None
    side = executor.__dict__.get('_flash_s5_decode_step')
    if side is None:
        side = module.EngineSide(logger=_progress_logger())
        executor._flash_s5_decode_step = side
    return side


def _decode_step_rpc(proc, method, kwargs):
    """Worker: pop the tag before dispatch. Returns a clean-window RPC state, or None."""
    global _DECODE_STEP_WORKER
    module = _decode_step()
    if module is None:
        return None
    if _DECODE_STEP_WORKER is None:
        with _WORKER_PROGRESS_LOCK:
            if _DECODE_STEP_WORKER is None:
                _DECODE_STEP_WORKER = module.WorkerSide(logger=_progress_logger())
    name = method if type(method) is str else 'callable'
    popped = _DECODE_STEP_WORKER.pop(proc, name, kwargs)
    if popped is None or popped[1] == module.TAG_NONE:
        return None
    state = _worker_progress(proc.rank)
    anatomy = _anatomy(proc) if state is not None else None
    if anatomy is None:
        return None
    if popped[1] == module.TAG_CLOSE:
        # the window closed on the engine: this rank's clean rows as handler lines; the RPC itself is not a row
        anatomy.flush(_progress_logger())
        _DECODE_STEP_WORKER.counters['flushes'] += 1
        return None
    anatomy.begin(popped[2], name)
    _DECODE_STEP_WORKER.counters['clean_rows'] += 1
    return _CleanRpc([popped[2], name, time.monotonic_ns(), None, False])


# decode-merge probe (FLASH_S5_SUBMIT_CPU=1, default off): thread CPU time beside the wall stamps of flash_s5_submit on
# this worker, cumulative, one 'flash-s5 submit-cpu' line per chunk release. None until first read, False when off.
_SUBMIT_CPU = None
SUBMIT_CPU_BUCKETS_MS = (5, 10, 20)


def _submit_cpu():
    global _SUBMIT_CPU
    if _SUBMIT_CPU is None:
        _SUBMIT_CPU = ({'n': 0, 'wall_ns': 0, 'cpu_ns': 0, 'max_wall_ns': 0, 'max_cpu_ns': 0,
                        'by_wall_ms': {str(b): [0, 0, 0] for b in SUBMIT_CPU_BUCKETS_MS + ('more',)}}
                       if os.environ.get('FLASH_S5_SUBMIT_CPU', '0') == '1' else False)
    return _SUBMIT_CPU or None


def _note_submit_cpu(rpc):
    """decode-merge probe: one flash_s5_submit's (wall, thread CPU); host clocks only, never raises."""
    try:
        acc = _submit_cpu()
        wall, cpu = rpc[3] - rpc[2], time.thread_time_ns() - rpc[5]
        acc['n'] += 1
        acc['wall_ns'] += wall
        acc['cpu_ns'] += cpu
        acc['max_wall_ns'] = max(acc['max_wall_ns'], wall)
        acc['max_cpu_ns'] = max(acc['max_cpu_ns'], cpu)
        key = next((str(b) for b in SUBMIT_CPU_BUCKETS_MS if wall <= b * 1_000_000), 'more')
        row = acc['by_wall_ms'][key]
        row[0] += 1
        row[1] += wall
        row[2] += cpu
    except Exception:  # noqa: BLE001 - a measurement never fails the RPC
        pass


def _progress_logger():
    global _progress_log
    if _progress_log is None:
        # vLLM's init_logger is the path s5_startup's INFO line is proven on.
        try:
            from vllm.logger import init_logger
        except ImportError:
            import logging
            _progress_log = logging.getLogger(__name__)
        else:
            _progress_log = init_logger(__name__)
    return _progress_log


def _worker_progress(rank):
    """(trace, pending ordered emits) for this worker process, or None."""
    global _WORKER_PROGRESS
    state = _WORKER_PROGRESS
    if state is None:
        with _WORKER_PROGRESS_LOCK:
            if _WORKER_PROGRESS is None:
                from collections import deque
                mode = progress_mode(os.environ)
                _WORKER_PROGRESS = ((ProgressTrace('worker', rank), deque())
                                    if mode is not None else False)
            state = _WORKER_PROGRESS
    return state or None


def _anatomy(proc):
    """This worker's handler-anatomy recorder, or None. Unset: never imports handler_anatomy."""
    global _ANATOMY
    state = _ANATOMY
    if state is None:
        with _WORKER_PROGRESS_LOCK:
            if _ANATOMY is None:
                if os.environ.get('FLASH_S5_HANDLER_ANATOMY', '') != '1':
                    _ANATOMY = False
                else:
                    try:
                        import handler_anatomy
                        _ANATOMY = handler_anatomy.install(proc)
                    except Exception as error:
                        # A measurement never fails the RPC: refuse once, visibly, and stay off.
                        _ANATOMY = False
                        _progress_logger().warning('flash-s5 handler install refused: %s', type(error).__name__)
            state = _ANATOMY
    return state or None


def _reply_lock(proc):
    lock = proc.__dict__.get('_flash_s5_reply_lock')
    if lock is None:
        with _REPLY_LOCKS_LOCK:
            lock = proc.__dict__.get('_flash_s5_reply_lock')
            if lock is None:
                lock = threading.Lock()
                proc._flash_s5_reply_lock = lock
    return lock


def worker_rpc_begin(proc, method, kwargs):
    """Pop the engine's progress meta before dispatch; stock methods never see it."""
    clean = _decode_step_rpc(proc, method, kwargs) if _DECODE_STEP is not False else None
    if type(kwargs) is not dict or PROGRESS_META_KEY not in kwargs:
        return clean
    sequence, direct = decode_progress_meta(kwargs.pop(PROGRESS_META_KEY))
    name = method if type(method) is str else 'callable'
    if direct and name not in PROGRESS_DIRECT_METHODS and not (
            name in PROGRESS_DIRECT_INWAIT_METHODS and os.environ.get(INWAIT_ENV, '0') == '1'):
        # Raised inside the RPC's try: the monitored worker-failure report and a
        # FAILURE wire, which the engine receiver poisons on before matching.
        raise ValueError('flash-s5 direct reply refused for this method')
    state = _worker_progress(proc.rank)
    if state is not None and name == 'flash_s5_release':
        # One receipt line per chunk: everything this rank recorded before the
        # chunk's release. Logging failures are counted by the trace, never raised.
        # Alpha-next PLAN 2: one atomic write per part through the process receipt writer.
        try:
            emit_receipt_lines(state[0].lines(), _progress_logger())
        except Exception:
            state[0].dropped_total += 1
        if (anatomy := _anatomy(proc)) is not None:
            anatomy.flush(_progress_logger())
        if (acc := _submit_cpu()) is not None:
            # decode-merge probe: cumulative submit wall/CPU of this rank, one line per chunk release.
            try:
                _progress_logger().info('flash-s5 submit-cpu ' + json.dumps(dict(acc, rank=proc.rank), sort_keys=True, separators=(',', ':')))
            except Exception:  # noqa: BLE001 - a measurement never fails the RPC
                pass
    if state is not None and (anatomy := _anatomy(proc)) is not None:
        anatomy.begin(sequence, name)
    if state is not None and name == 'flash_s5_submit' and _submit_cpu() is not None:
        # decode-merge probe: thread CPU stamp beside the wall stamp (rpc[5]).
        return [sequence, name, time.monotonic_ns(), None, direct, time.thread_time_ns()]
    return [sequence, name, time.monotonic_ns(), None, direct]


def worker_rpc_returned(proc, rpc, output_rank):
    if rpc is None:
        return
    rpc[3] = time.monotonic_ns()
    if len(rpc) > 5:
        _note_submit_cpu(rpc)       # decode-merge probe (FLASH_S5_SUBMIT_CPU=1 only)
    if _ANATOMY:
        _ANATOMY.returned(rpc, output_rank)
    if output_rank is None or proc.rank == output_rank:
        return
    state = _worker_progress(proc.rank)
    if state is not None and type(rpc) is not _CleanRpc:
        state[0].record('r', rpc[0], rpc[1], rpc[2], rpc[3], None, int(rpc[4]), 0)


_DIRECT_WRITE = threading.local()


def worker_handle_output(proc, rpc, output):
    """Route one reply: direct (typed native command flow) or stock-ordered."""
    if output is STAGE_SKIP:
        # A stage-scoped command for other ranks: no reply is expected or written.
        state = None if rpc is None else _worker_progress(proc.rank)
        if state is not None:
            state[0].record('r', rpc[0], rpc[1], rpc[2], rpc[3], None, int(rpc[4]), 0)
        return
    if rpc is not None and rpc[4]:
        # Written by this RPC thread now, not queued behind async decode output.
        # A raised command still goes through stock enqueue_output, which turns
        # it into a FAILURE wire; the engine receiver poisons on any non-SUCCESS
        # status before matching, so the failure is prompt and never correlated.
        _DIRECT_WRITE.active = True
        try:
            proc.enqueue_output(output if isinstance(output, Exception) else DirectReply(rpc[0], output))
        finally:
            _DIRECT_WRITE.active = False
        state = _worker_progress(proc.rank)
        if state is not None:
            state[0].record('r', rpc[0], rpc[1], rpc[2], rpc[3], time.monotonic_ns(), 1, 1)
        return
    state = _worker_progress(proc.rank)
    if state is not None:
        # Ordered replies leave in exactly handle_output order (async FIFO).
        state[1].append(rpc)
        if rpc is not None and (anatomy := _anatomy(proc)) is not None:
            return anatomy.handle_output(proc, rpc, output)
    proc.handle_output(output)


def worker_enqueue_reply(proc, result):
    """The stock response-MQ write, serialized against a direct writer thread."""
    with _reply_lock(proc):
        if (response_mq := proc.worker_response_mq) is not None:
            response_mq.enqueue(result)
    state = _worker_progress(proc.rank)
    if state is None or getattr(_DIRECT_WRITE, 'active', False):
        return
    try:
        rpc = state[1].popleft()
    except IndexError:
        return
    if rpc is not None:
        if type(rpc) is not _CleanRpc:
            state[0].record('r', rpc[0], rpc[1], rpc[2], rpc[3], time.monotonic_ns(), 0, 1)
        if (anatomy := _anatomy(proc)) is not None:
            anatomy.replied(rpc)


__all__ = [
    'BoundedSender', 'CORE_PATH', 'CORE_SHA256', 'EXECUTOR_PATH',
    'EXECUTOR_SHA256', 'LocalFuture', 'PINS', 'WORKER_PATH', 'WORKER_SHA256',
    'after_engine_startup', 'before_abort', 'before_engine_shutdown',
    'before_finish_pause', 'before_pause', 'before_reset_prefix',
    'before_shutdown', 'before_step_with_batch_queue', 'collective_rpc',
    'config_reply_timeout', 'control_pending', 'engine_idle_tick', 'executor_shutdown',
    'native_progress_wait',
    'generate_core', 'generate_executor', 'engine_request_admitted',
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
# payloads are normally omitted). Native dispatcher quarantine is the explicit
# diagnostic exception: bounded cause text and frame locations, without locals.
FATAL_OPTIONAL = {'detail': 160, 'where': 100, 'cause': 80,
                  'native_message': 160, 'native_trace': 320}


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
            # Native quarantine wraps the useful device error; do not lose it
            # when this worker parks before its normal exception logger runs.
            if type(error).__name__ == 'LaneDispatchError' and str(error).startswith('dispatcher quarantined:'):
                root, seen = error, set()
                while id(root) not in seen and len(seen) < 8:
                    seen.add(id(root))
                    linked = root.__cause__ if root.__cause__ is not None else root.__context__
                    if linked is None or id(linked) in seen:
                        break
                    root = linked
                rows['native_message'] = type(root).__name__ + ': ' + str(root)
                frames, tb = [], root.__traceback__
                while tb is not None:
                    code = tb.tb_frame.f_code
                    frames.append('%s:%d:%s' % (os.path.basename(code.co_filename), tb.tb_lineno, code.co_name))
                    tb = tb.tb_next
                # No locals, source lines, tensor contents or absolute paths.
                rows['native_trace'] = ' <- '.join(frames[-8:])[-FATAL_OPTIONAL['native_trace']:]
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
            # Escape-heavy device messages must still fit the existing pipe.
            for key in ('native_trace', 'native_message'):
                while len(data) > FATAL_RECORD_MAX and len(record.get(key, '')) > 32:
                    record[key] = record[key][32:] if key == 'native_trace' else record[key][:-32]
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
                         'where=%s detail=%r native_message=%r native_trace=%r; '
                         'executor shutdown follows (worker fatal channel)',
                         record.get('rank'), record.get('pid'), record.get('method'), record.get('error_type'),
                         record.get('code'), record.get('cause'), record.get('where'), record.get('detail'),
                         record.get('native_message'), record.get('native_trace'))
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


def engine_request_admitted(engine, request):
    """Timestamp scheduler acceptance, not the client's pre-tokenization send."""
    try:
        driver = getattr(engine, '_flash_s5_driver', None)
        emit = getattr(driver, '_gap_event', None)
        if callable(emit):
            # night-pack fix 3: the API process's arrival stamp (wall clock) and the admission's wall clock beside it,
            # so arrival -> admit reads in one clock domain and splits the sent -> admit constant (log-only)
            emit('request_admitted', request_uid=request.request_id,
                 prompt_tokens=request.num_prompt_tokens, arrival_s=getattr(request, 'arrival_time', None),
                 admit_wall_s=time.time())
    except Exception:
        pass
    try:                                # ingress prefix counter: log-only, engine arms only (the driver decides)
        note = getattr(getattr(engine, '_flash_s5_driver', None), 'note_ingress_prefix', None)
        if callable(note):
            note(engine, request)
    except Exception:
        pass


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
