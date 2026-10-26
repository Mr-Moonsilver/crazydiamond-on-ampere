"""E3 split decode graph bank and route transitions (D4, worker/native side).

Captures the pinned runner's FULL decode descriptors a second time inside the
green decode context, on the green decode stream, into a separate graph pool,
through a second instance of the stock ``ModelCudaGraphManager`` class. The
stock manager and its full-GPU bank stay untouched. ``cudagraph_utils`` is
transformed (``worker_hooks``) so that a manager carrying
``_flash_s5_capture_context`` captures on that stream instead of a fresh one.

Route transitions (full <-> split) happen only at overlap-phase boundaries:
close new submissions, record tail events on every decode consumer stream
(main, output copy, PP broadcast), wait for them, then swap the runner's and PP
handler's cached main stream and the active bank atomically. The route
generation increments on every switch; the storage epoch does not.

Evidence boundary: measured bytes are the free-memory delta around capture on
this rank; SM counts come from ``green_context``. Fake-torch tests exercise the
call paths only; node containment and cross-context progress are GPU checks.
"""
import time
from contextlib import contextmanager, nullcontext

from e3_types import E3ContractError
from graph_router import ExecutionContext, ExecutionDescriptor


class SplitBankError(E3ContractError):
    pass


FULL_MODE = 'FULL'


def _torch():
    import torch

    return torch


def _mode_named(enum_or_dict_keys, name):
    for mode in enum_or_dict_keys:
        if getattr(mode, 'name', None) == name:
            return mode
    return None


def full_descriptors(stock_manager):
    """The stock FULL descriptors, in capture order, with their local mirrors."""
    descs = getattr(stock_manager, '_capture_descs', None)
    if not isinstance(descs, dict):
        raise SplitBankError('stock cudagraph manager has no capture descriptors')
    mode = _mode_named(descs.keys(), FULL_MODE)
    if mode is None or not descs[mode]:
        raise SplitBankError('stock cudagraph manager captured no FULL descriptors')
    pairs = []
    for desc in descs[mode]:
        pairs.append((desc, ExecutionDescriptor.from_vllm(desc)))
    return mode, tuple(pairs)


class SplitGraphHandle:
    """Opaque handle for one captured FULL descriptor in the split bank."""

    __slots__ = ('descriptor', 'local', 'bank', 'replays', 'destroyed')

    def __init__(self, descriptor, local, bank):
        self.descriptor = descriptor
        self.local = local
        self.bank = bank
        self.replays = 0
        self.destroyed = False


class SplitDecodeBank:
    """Second manager instance, own pool, captured on the green decode stream."""

    def __init__(self, runner, split, *, torch_module=None, clock=time.monotonic):
        self.torch = torch_module if torch_module is not None else _torch()
        self.runner = runner
        self.split = split
        self.clock = clock
        stock = getattr(runner, 'cudagraph_manager', None)
        if stock is None:
            raise SplitBankError('runner has no stock cudagraph manager')
        if getattr(runner, 'lora_config', None) is not None:
            raise SplitBankError('split bank does not support LoRA capture')
        self.stock = stock
        self.full_mode, self.pairs = full_descriptors(stock)
        self.manager = None
        self.pool = None
        self.captured = False
        self.measured_bytes = None
        self.capture_seconds = None
        self.capture_memory = None
        self.handles = {}
        self._context_factory = None
        self.capture_streams = None
        self.capture_failed = False

    # -- construction -----------------------------------------------------
    def build_manager(self):
        if self.manager is not None:
            raise SplitBankError('split manager already built')
        stock = self.stock
        cls = type(stock)
        manager = cls(self.runner.vllm_config, self.runner.device, stock.cudagraph_mode,
                      decode_query_len=stock.decode_query_len,
                      lora_capture_cases=getattr(stock, 'lora_capture_cases', None),
                      varlen_decode=getattr(stock, 'varlen_decode', False))
        pool = self.torch.cuda.graph_pool_handle()
        if pool == getattr(stock, 'pool', None):
            raise SplitBankError('split bank must not share the stock graph pool')
        manager.pool = pool
        manager._capture_descs = {self.full_mode: [desc for desc, _ in self.pairs]}
        decode_stream = self.split.decode_stream
        if decode_stream is None:
            raise SplitBankError('green split has no decode stream')

        def context_factory():
            from vllm.distributed.parallel_state import GraphCaptureContext

            return GraphCaptureContext(decode_stream)

        self._context_factory = context_factory
        manager._flash_s5_capture_context = context_factory
        manager._flash_s5_capture_owner = self.split
        self.manager = manager
        self.pool = pool
        return manager

    def capture_kwargs(self):
        """Mirror the stock ``capture_model`` call for the split manager."""
        runner = self.runner
        input_buffers = runner.input_buffers
        pcp = getattr(runner, 'pcp_manager', None)
        if pcp is not None:
            input_buffers = pcp.input_buffers
        return dict(model=runner.model, model_state=runner.model_state,
                    input_buffers=input_buffers,
                    intermediate_tensors=runner.intermediate_tensors,
                    block_tables=runner.block_tables, attn_groups=runner.attn_groups,
                    kv_cache_config=runner.kv_cache_config, pcp_manager=pcp,
                    has_lora=False,
                    use_aux_hidden_state_outputs=getattr(runner, 'use_aux_hidden_state_outputs', False),
                    lora_capture_hook=None,
                    progress_bar_desc='Capturing E3 split decode graphs')

    def capture(self, *, set_graph_pool_id=None, memory_info=None):
        """Capture every FULL descriptor once; measure the pool delta on this rank."""
        if self.captured or self.capture_failed:
            raise SplitBankError('split bank captured or quarantined')
        if self.manager is None:
            self.build_manager()
        torch = self.torch
        if memory_info is None:
            memory_info = torch.accelerator.get_memory_info
        torch.accelerator.synchronize()
        free_before = int(memory_info()[0])
        started = self.clock()
        try:
            from capture_streams import for_runner
            self.capture_streams = for_runner(self.runner, self.split)
            with self.capture_streams.capture():
                self.manager.capture(**self.capture_kwargs())
        except BaseException:
            self.capture_failed = True
            self.captured = False
            raise
        finally:
            if self.capture_streams is not None and self.capture_streams.retained:
                for graph in self.manager.graphs.values(): self.capture_streams.attach_graph(graph)
            # The stock capture pointed the pynccl allocator at our pool;
            # the stock manager's pool must stay the process default afterwards.
            if set_graph_pool_id is not None:
                set_graph_pool_id(self.stock.pool)
        torch.accelerator.synchronize()
        free_after = int(memory_info()[0])
        self.capture_seconds = self.clock() - started
        self.measured_bytes = free_before - free_after
        self.capture_memory = {
            'free_before_bytes': free_before, 'free_after_bytes': free_after,
            'delta_bytes': self.measured_bytes,
            'basis': 'device-free samples around synchronized capture, not allocator pool attribution',
        }
        graphs = getattr(self.manager, 'graphs', {})
        for desc, local in self.pairs:
            if desc not in graphs:
                raise SplitBankError('split capture left a FULL descriptor without a graph')
            self.handles[local] = SplitGraphHandle(desc, local, self)
        self.captured = True
        return self.measured_bytes

    # -- B backend protocol -----------------------------------------------
    def handle_for(self, local):
        if type(local) is not ExecutionDescriptor:
            raise SplitBankError('split bank lookup needs an ExecutionDescriptor')
        handle = self.handles.get(local)
        if handle is None:
            raise SplitBankError('descriptor is not captured in the split bank')
        return handle

    def replay(self, handle):
        if not isinstance(handle, SplitGraphHandle) or handle.bank is not self:
            raise SplitBankError('replay needs a handle of this split bank')
        if handle.destroyed:
            raise SplitBankError('replay of a destroyed split graph')
        output = self.manager.run_fullgraph(handle.descriptor)
        handle.replays += 1
        return output

    def destroy(self, handle):
        if not isinstance(handle, SplitGraphHandle) or handle.bank is not self:
            raise SplitBankError('destroy needs a handle of this split bank')
        if handle.destroyed:
            raise SplitBankError('split graph already destroyed')
        if self.capture_streams is not None: self.capture_streams.drain()
        graph = self.manager.graphs.pop(handle.descriptor, None)
        if graph is not None and callable(getattr(graph, 'reset', None)):
            graph.reset()
            if self.capture_streams is not None: self.capture_streams.retire_graph(graph)
        handle.destroyed = True
        del self.handles[handle.local]
        return True

    def execution_context(self, name='decode-split'):
        """B's ExecutionContext for this bank; bytes measured after capture."""
        split = self.split
        if split.decode_sms is None:
            raise SplitBankError('green split is not created')
        if self.manager is None:
            self.build_manager()
        reserved = self.measured_bytes if self.measured_bytes is not None else 1
        basis = 'measured' if self.measured_bytes is not None else 'estimate'
        return ExecutionContext(name, split.decode_context_id,
                                requested_sm=split.requested_decode_sms,
                                measured_sm=split.decode_sms, pool_id=self._pool_id(),
                                stream_token=split.decode_stream_token,
                                reserved_bytes=max(int(reserved), 1), basis=basis)

    def _pool_id(self):
        pool = self.pool
        if pool is None:
            raise SplitBankError('split pool is not created')
        if isinstance(pool, tuple):
            return int(pool[0]) * 1000003 + int(pool[1])
        return int(pool) if type(pool) is int else id(pool)

    def descriptors(self):
        return tuple(local for _, local in self.pairs)


class SplitCaptureBackend:
    """Adapter from B's per-descriptor backend protocol to one bank capture."""

    native_verified = False

    def __init__(self, bank, *, set_graph_pool_id=None):
        if not isinstance(bank, SplitDecodeBank):
            raise SplitBankError('capture backend needs a SplitDecodeBank')
        self.bank = bank
        self.set_graph_pool_id = set_graph_pool_id
        self._events = set()
        self._streams = set()

    def capture(self, bank_name, key, lane, context):
        bank = self.bank
        if context.stream_token != bank.split.decode_stream_token:
            raise SplitBankError('bank context names another stream than the green decode stream')
        if context.context_id != bank.split.decode_context_id:
            raise SplitBankError('bank context names another green context')
        if not bank.captured:
            bank.capture(set_graph_pool_id=self.set_graph_pool_id)
        return bank.handle_for(key.descriptor)

    def replay(self, handle, key, lane, context):
        bank = self.bank
        if not isinstance(handle, SplitGraphHandle):
            raise SplitBankError('replay needs a split graph handle')
        if handle.local != key.descriptor:
            raise SplitBankError('replay key names another descriptor')
        current = bank.torch.cuda.current_stream(bank.runner.device)
        token = int(getattr(current, 'cuda_stream', -1))
        if token != context.stream_token:
            raise SplitBankError('split replay must run with the green decode stream current')
        return bank.replay(handle)

    def destroy_graph(self, handle):
        return self.bank.destroy(handle)

    def destroy_event(self, token):
        if id(token) not in self._events:
            raise SplitBankError('destroy of an unregistered event')
        self._events.discard(id(token))

    def destroy_stream(self, token):
        if id(token) not in self._streams:
            raise SplitBankError('destroy of an unregistered stream')
        self._streams.discard(id(token))


# --- route transitions -----------------------------------------------------

class TransitionRecord:
    __slots__ = ('generation', 'source', 'target', 'drained_streams', 'seconds')

    def __init__(self, generation, source, target, drained_streams, seconds):
        self.generation = generation
        self.source = source
        self.target = target
        self.drained_streams = drained_streams
        self.seconds = seconds


class RouteController:
    """Worker-side full/split route owner for one runner.

    ``route_generation`` increments on each switch. While ``route == 'split'``
    the runner's cached ``main_stream`` and the PP handler's ``main_stream``
    are the green decode stream, and ``decode_scope`` makes it current for the
    whole worker ``execute_model``/``sample_tokens`` call.
    """

    def __init__(self, runner, bank, *, torch_module=None, clock=time.monotonic):
        if not isinstance(bank, SplitDecodeBank):
            raise SplitBankError('route controller needs a SplitDecodeBank')
        self.torch = torch_module if torch_module is not None else _torch()
        self.runner = runner
        self.bank = bank
        self.clock = clock
        self.route = 'full'
        self.route_generation = 0
        self.transitions = []
        self.full_main_stream = runner.main_stream
        self.full_pp_stream = self._pp_main_stream()
        self.admitting = True
        runner._flash_s5_decode_stream = None
        runner._flash_s5_route = self
        runner._flash_s5_torch = self.torch

    def _pp_handler(self):
        return getattr(self.runner, 'pp_handler', None)

    def _pp_main_stream(self):
        handler = self._pp_handler()
        return getattr(handler, 'main_stream', None) if handler is not None else None

    def consumer_streams(self):
        """Every stream that consumes decode work and must be drained."""
        streams = [self.runner.main_stream]
        if self.bank.capture_streams is not None:
            streams.extend(self.bank.capture_streams.consumer_streams())
        for name in ('output_copy_stream',):
            stream = getattr(self.runner, name, None)
            if stream is not None:
                streams.append(stream)
        handler = self._pp_handler()
        if handler is not None:
            for name in ('main_stream', 'broadcast_stream'):
                stream = getattr(handler, name, None)
                if stream is not None:
                    streams.append(stream)
        unique = []
        for stream in streams:
            if all(stream is not seen for seen in unique):
                unique.append(stream)
        return unique

    def drain(self):
        """Record a tail event on each consumer stream and wait for all."""
        events = []
        for stream in self.consumer_streams():
            event = self.torch.cuda.Event(enable_timing=False)
            event.record(stream)
            events.append(event)
        for event in events:
            event.synchronize()
        return len(events)

    def _switch(self, target):
        if target not in ('full', 'split'):
            raise SplitBankError(f'unknown route {target!r}')
        if target == self.route:
            raise SplitBankError(f'route is already {target!r}')
        if target == 'split' and not self.bank.captured:
            raise SplitBankError('split route needs a captured split bank')
        started = self.clock()
        self.admitting = False
        try:
            drained = self.drain()
            if target == 'split':
                stream = self.bank.split.decode_stream
                if stream is None:
                    raise SplitBankError('green split has no decode stream')
                self.runner.__dict__['main_stream'] = stream
                handler = self._pp_handler()
                if handler is not None:
                    handler.main_stream = stream
                self.runner._flash_s5_decode_stream = stream
            else:
                self.runner.__dict__['main_stream'] = self.full_main_stream
                handler = self._pp_handler()
                if handler is not None:
                    handler.main_stream = self.full_pp_stream
                self.runner._flash_s5_decode_stream = None
            source = self.route
            self.route = target
            self.route_generation += 1
            record = TransitionRecord(self.route_generation, source, target, drained,
                                      self.clock() - started)
            self.transitions.append(record)
            return record
        finally:
            self.admitting = True

    def enter_split(self):
        return self._switch('split')

    def leave_split(self):
        return self._switch('full')

    def decode_stream(self):
        return self.runner._flash_s5_decode_stream


def _runner_torch(runner):
    torch = getattr(runner, '_flash_s5_torch', None)
    return torch if torch is not None else _torch()


def decode_scope(runner):
    """Make the green decode stream current for one worker decode call."""
    stream = getattr(runner, '_flash_s5_decode_stream', None)
    if stream is None:
        return nullcontext()
    return _runner_torch(runner).cuda.stream(stream)


@contextmanager
def prefill_scope(runner):
    """Make the green prefill stream current for one bounded lane operation."""
    split = getattr(runner, '_flash_s5_split', None)
    if split is None or split.prefill_stream is None:
        raise SplitBankError('prefill scope needs an open green split')
    with _runner_torch(runner).cuda.stream(split.prefill_stream):
        yield split.prefill_stream


def install(runner, split, *, torch_module=None, set_graph_pool_id=None, clock=time.monotonic):
    """Build the split bank objects on a runner that already has its stock manager.

    Capture itself happens through B's ``capture_banks`` seam after the stock
    capture (the backend captures the whole bank on its first call).
    """
    if getattr(runner, '_flash_s5_split_bank', None) is not None:
        raise SplitBankError('split bank already installed')
    bank = SplitDecodeBank(runner, split, torch_module=torch_module, clock=clock)
    runner._flash_s5_torch = bank.torch
    bank.build_manager()
    backend = SplitCaptureBackend(bank, set_graph_pool_id=set_graph_pool_id)
    controller = RouteController(runner, bank, torch_module=torch_module, clock=clock)
    runner._flash_s5_split = split
    runner._flash_s5_split_bank = bank
    runner._flash_s5_split_backend = backend
    runner._flash_s5_graph_capture = backend.capture
    runner._flash_s5_split_descriptors = bank.descriptors()
    return bank, backend, controller


def shutdown(runner):
    """Destroy graphs and logically close; exposed native contexts live to exit."""
    controller = getattr(runner, '_flash_s5_route', None)
    if controller is not None and controller.route != 'full':
        raise SplitBankError('shutdown requires the full route (leave split first)')
    bank = getattr(runner, '_flash_s5_split_bank', None)
    if bank is not None:
        if controller is not None: controller.drain()
        for handle in list(bank.handles.values()):
            bank.destroy(handle)
        # Also reset graphs retained by a manager whose capture failed before
        # SplitGraphHandles were published. Never release auxiliary streams first.
        if bank.capture_streams is not None: bank.capture_streams.drain()
        if bank.manager is not None:
            for graph in list(bank.manager.graphs.values()):
                graph.reset()
                if bank.capture_streams is not None: bank.capture_streams.retire_graph(graph)
            bank.manager.graphs.clear()
        if bank.capture_streams is not None: bank.capture_streams.close_after_graphs()
        bank.manager = None
        bank.pool = None
    split = getattr(runner, '_flash_s5_split', None)
    if split is not None and split.state == 'open':
        split.close()
    return True


__all__ = ['RouteController', 'SplitBankError', 'SplitCaptureBackend', 'SplitDecodeBank',
           'SplitGraphHandle', 'TransitionRecord', 'decode_scope', 'full_descriptors',
           'install', 'prefill_scope', 'shutdown']
