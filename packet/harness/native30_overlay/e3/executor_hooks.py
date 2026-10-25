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

from e3_types import E3ContractError, E3Pending
from reply_transport import Backpressure, NotReady, ReplyTransport, TransportError


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

    def flash_s5_cancel(self, message):
        return _flash_s5_hooks.worker_rpc(self, "cancel", message)

    def flash_s5_transition(self, message):
        return _flash_s5_hooks.worker_rpc(self, "transition", message)

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
    ('''                for w in workers:
                    # Shutdown response queues
''', '''                _flash_s5_hooks.before_shutdown(self)
                for w in workers:
                    # Shutdown response queues
'''),
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
     '        _flash_s5_hooks.before_shutdown(self)\n'
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

_CORE_FORWARD = (
    ('logger = init_logger(__name__)\n', 'logger = init_logger(__name__)\n\n' + _IMPORT),
    ('''        self.is_ec_consumer = (
''', '''        _flash_s5_hooks.after_engine_startup(self)
        self.is_ec_consumer = (
'''),
    ('''        batch_queue = self.batch_queue
        assert batch_queue is not None
''', '''        batch_queue = self.batch_queue
        assert batch_queue is not None
        _flash_s5_ready = _flash_s5_hooks.before_step_with_batch_queue(self)
        if _flash_s5_ready is not None:
            return _flash_s5_ready
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
        self._consumed = False
        self._taken = None
        self._taken_lag = None
        self._result_ready = False
        self._result = None

    def _transport(self):
        transport = getattr(self.executor, '_flash_s5_transport', None)
        if transport is None:
            raise TransportError('owned reply transport is not installed')
        return transport

    def poll(self, budget=1):
        transport = self._transport()
        transport.poll(budget)
        call = transport._calls.get(self.sequence)
        return bool(call is not None and call.ready)

    def done(self):
        if self._consumed:
            return True
        return self.poll(1)

    def take(self):
        if self._consumed:
            raise NotReady('future result already consumed')
        value, lag = self._transport().take(self.sequence)
        self._consumed = True
        self._taken, self._taken_lag = value, lag
        return value

    def result(self, timeout=None):
        if self._result_ready:
            return self._result
        if self._consumed:
            self._result = self.aggregate(self._taken) if self.aggregate else self._taken
            self._result_ready = True
            return self._result
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            transport = self._transport()
            transport.poll(1)
            call = transport._calls.get(self.sequence)
            if call is None:
                raise TransportError('reply expectation disappeared')
            if call.ready:
                value = self.take()
                self._result = self.aggregate(value) if self.aggregate else value
                self._result_ready = True
                return self._result
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('reply not ready before deadline')
            time.sleep(0.001)


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


def install(executor, *, config, success_status, clock=time.monotonic,
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
    def send(sequence):
        executor.rpc_broadcast_mq.enqueue((send_method, args, kwargs,
                                           broadcast_rank))
    sequence = transport.submit(kind, send, ranks=ranks,
                                timeout_s=effective_timeout, aggregate=None)
    future = LocalFuture(executor, sequence, aggregate=aggregate)
    return future if non_block else future.result(timeout=effective_timeout)


def config_reply_timeout(executor):
    """The configured reply budget is carried on the owner; default is finite."""
    handoff = getattr(executor, '_flash_s5_handoff', None)
    if handoff is not None:
        return handoff.config.reply_timeout_s
    return 300.0


def before_shutdown(executor):
    """Close the owned receiver before response MQs are destroyed.

    Returns ``(clean, report)``. An unclean close keeps the transport and its
    MQs owned; ``retain_mq`` then refuses to tear those queues down. No forced
    thread cancellation or lease retirement is implied.
    """
    transport = getattr(executor, '_flash_s5_transport', None)
    if transport is None:
        return True, None
    close = getattr(executor, '_flash_s5_close', None)
    if close is not None:
        return bool(close.get('clean')), close
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
        return driver.advance()
    handoff = _handoff(engine)
    if handoff is None:
        return None
    report = handoff.poll(8)
    if report.failed:
        raise E3ContractError('E3 handoff is quarantined')
    return None


def before_abort(engine, request_ids):
    driver = _driver(engine)
    if driver is not None:
        if hasattr(driver, 'tx'):
            # Native driver: abort may target a begun transaction or a chunk
            # that is only parked in the admission controller.
            transaction = driver.tx
            if transaction is None and driver.controller.active:
                transaction = driver.controller.admission().transaction
            if (transaction is not None
                    and transaction.request_uid in tuple(request_ids)):
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

def worker_rpc(worker, name, message):
    """Generated worker entry point for typed open/submit/poll/commit/cancel."""
    if name not in ('open', 'submit', 'poll', 'commit', 'cancel', 'transition'):
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
    'config_reply_timeout', 'generate_core', 'generate_executor',
    'generate_worker', 'install', 'park_prefill', 'retain_mq', 'revert_core',
    'revert_executor', 'revert_worker', 'worker_rpc',
]
