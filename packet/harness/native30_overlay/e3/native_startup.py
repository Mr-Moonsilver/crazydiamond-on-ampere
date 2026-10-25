"""E3 worker-native startup composition and the transition RPC (D4).

Called once per GPU worker from the runner's E3 startup, after the stock
cudagraph manager exists and before capture. It needs the green split created
at worker init (``worker_hooks.after_runner_init``) and composes, in order:

1. the split decode bank objects (capture happens later through B's
   ``capture_banks`` seam);
2. the prefill-lane communicators (PyNccl TP, tagged PP channel), warmed in a
   fixed rank order on the prefill stream;
3. the lane-private PLE prefetch scratch for the one PLE layer on this stage;
4. a ``LaneForward`` factory for the worker operation server;
5. the graph-router builder that registers the split bank with B.

``transition_rpc`` is the worker side of ``flash_s5_transition``: it validates
the engine's command, performs the drained stream/bank switch through the
``RouteController`` and acknowledges only afterwards with the new route
generation. No per-step ticket.

Everything here records bytes it allocated (PP buffers, PLE scratch, measured
split pool after capture) as an inventory; nothing claims funding.
"""
import time

from e3_types import E3ContractError
from lane_comm import PrefillCommunicators
from model_hooks import LaneForward, PlePrivateScratch, ple_embedding
from split_bank import install as install_split_bank

SPLIT_BANK_NAME = 'decode-split'
FULL_ROUTE = 'full'
SPLIT_ROUTE = 'split'


class NativeStartupError(E3ContractError):
    pass


def _torch():
    import torch

    return torch


def _groups(tp_group, pp_group):
    if tp_group is not None and pp_group is not None:
        return tp_group, pp_group
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    return (tp_group if tp_group is not None else get_tp_group(),
            pp_group if pp_group is not None else get_pp_group())


def _model_dtype(runner, model):
    config = getattr(getattr(runner, 'vllm_config', None), 'model_config', None)
    dtype = getattr(config, 'dtype', None)
    if dtype is not None:
        return dtype
    parameters = getattr(model, 'parameters', None)
    if callable(parameters):
        for parameter in parameters():
            return parameter.dtype
    raise NativeStartupError('cannot discover the model dtype for the PP channel')


def _hc_width(model):
    config = getattr(model, 'config', None)
    hc = getattr(config, 'hc_count', None)
    hidden = getattr(config, 'hidden_size', None)
    if type(hc) is not int or type(hidden) is not int or hc < 1 or hidden < 1:
        raise NativeStartupError('model config lacks hc_count/hidden_size for the PP channel')
    return hc * hidden


def _ple_scratch(model, chunk_tokens, torch, device):
    found = []
    for layer_idx in range(int(model.start_layer), int(model.end_layer)):
        embedding = ple_embedding(model, layer_idx)
        if embedding is not None:
            found.append((layer_idx, embedding))
    if not found:
        return None, None
    if len(found) > 1:
        raise NativeStartupError('more than one PLE layer on this stage is not supported')
    layer_idx, embedding = found[0]
    return layer_idx, PlePrivateScratch(embedding, chunk_tokens, torch_module=torch, device=device)


class WorkerNative:
    """Everything the lane needs on one worker, with its byte inventory."""

    def __init__(self, runner, split, bank, backend, controller, comms, ple_layer, ple_scratch,
                 chunk_tokens, tp_group, torch):
        self.runner = runner
        self.split = split
        self.bank = bank
        self.backend = backend
        self.controller = controller
        self.comms = comms
        self.ple_layer = ple_layer
        self.ple_scratch = ple_scratch
        self.chunk_tokens = chunk_tokens
        self.tp_group = tp_group
        self.torch = torch
        self.host_submit_ns = []

    def lane_forward(self, inputs, *, forward_context=None, attention_driver=None, clock=None):
        kwargs = dict(stream=self.split.prefill_stream, tp_group=self.tp_group,
                      tp_comm=self.comms.tp, forward_context=forward_context,
                      ple_scratch=self.ple_scratch, attention_driver=attention_driver,
                      torch_module=self.torch)
        if clock is not None:
            kwargs['clock'] = clock
        return LaneForward(self.runner.model, inputs, **kwargs)

    def inventory(self):
        record = {
            'split': self.split.describe(),
            'split_bank_bytes': self.bank.measured_bytes,
            'split_bank_basis': 'measured' if self.bank.measured_bytes is not None else 'uncaptured',
            'pp_channel_bytes': self.comms.pp.bytes_owned() if self.comms.pp is not None else 0,
            'ple_scratch_bytes': self.ple_scratch.bytes if self.ple_scratch is not None else 0,
            'ple_layer': self.ple_layer,
            'communicators': 'unmeasured',
            'chunk_tokens': self.chunk_tokens,
            'route': self.controller.route,
            'route_generation': self.controller.route_generation,
        }
        return record

    def graph_builder(self, *, router_class=None, full_context_id, full_pool_id,
                      full_allocated_bytes, engine_epoch=None):
        """B's ``runner._flash_e3_graph_builder``: register the split bank once."""
        if router_class is None:
            from graph_router import GraphRouter as router_class

        def build(current):
            if getattr(current, '_flash_e3_graph_builder_used', False):
                raise NativeStartupError('E3 graph builder was already used')
            lane = getattr(current, '_flash_e3_lane', None)
            state = getattr(lane, 'state', None)
            handle = getattr(getattr(state, 'decode', None), 'handle', None)
            if state is None or handle is None:
                raise NativeStartupError('graph builder needs A\'s worker state with a decode handle')
            compilation = getattr(current, '_flash_s5_compilation', None)
            if compilation is None:
                raise NativeStartupError('graph builder needs the compilation identity')
            router = router_class(state, handle, compilation, engine_epoch=engine_epoch,
                                  backend=self.backend, full_context_id=full_context_id,
                                  full_pool_id=full_pool_id,
                                  full_allocated_bytes=full_allocated_bytes,
                                  full_basis='measured')
            context = self.bank.execution_context(SPLIT_BANK_NAME)
            router.add_bank(SPLIT_BANK_NAME, context, self.bank.descriptors())
            split = self.split
            torch = self.torch

            def probe():
                stream = torch.cuda.current_stream(self.runner.device)
                return (split.decode_context_id, int(getattr(stream, 'cuda_stream', -1)))

            router.probe = probe
            current._flash_e3_graph_builder_used = True
            current._flash_e3_graph_router_split = self
            return router

        return build


def install_worker_native(runner, *, chunk_tokens, env=None, torch_module=None,
                          pynccl_class=None, set_graph_pool_id=None, tp_group=None,
                          pp_group=None, clock=time.monotonic):
    """Compose the worker-native lane objects on a runner with its stock manager."""
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise NativeStartupError('chunk_tokens must be a positive int')
    if getattr(runner, '_flash_s5_native', None) is not None and isinstance(
            runner._flash_s5_native, WorkerNative):
        raise NativeStartupError('worker-native lane already installed')
    split = getattr(runner, '_flash_s5_split', None)
    if split is None or getattr(split, 'state', None) != 'open':
        raise NativeStartupError('worker-native startup needs the open green split from worker init')
    torch = torch_module if torch_module is not None else _torch()
    model = getattr(runner, 'model', None)
    if model is None:
        raise NativeStartupError('runner has no model')
    tp_group, pp_group = _groups(tp_group, pp_group)

    bank, backend, controller = install_split_bank(
        runner, split, torch_module=torch, set_graph_pool_id=set_graph_pool_id, clock=clock)
    comms = PrefillCommunicators.build(
        tp_group, pp_group, runner.device, split.prefill_stream, max_tokens=chunk_tokens,
        hidden_width=_hc_width(model), dtype=_model_dtype(runner, model),
        pynccl_class=pynccl_class, torch_module=torch)
    comms.warm(split.prefill_stream)
    ple_layer, ple_scratch = _ple_scratch(model, chunk_tokens, torch, runner.device)
    native = WorkerNative(runner, split, bank, backend, controller, comms, ple_layer, ple_scratch,
                          chunk_tokens, tp_group, torch)
    runner._flash_s5_native = native
    runner._flash_s5_comms = comms
    runner._flash_s5_ple_scratch = ple_scratch
    runner._flash_s5_lane_forward = native.lane_forward
    runner._flash_s5_inventory = native.inventory
    return native


# --- transition RPC ---------------------------------------------------------------

_REQUIRED_COMMAND = ('engine_epoch', 'expected_route_generation', 'route_generation', 'target')


def transition_rpc(runner, command, *, engine_epoch=None, clock=time.monotonic):
    """Worker side of ``flash_s5_transition``; ACK only after the drained switch."""
    native = getattr(runner, '_flash_s5_native', None)
    if not isinstance(native, WorkerNative):
        raise NativeStartupError('transition before worker-native startup')
    if not isinstance(command, dict):
        raise NativeStartupError('transition command must be a mapping')
    for key in _REQUIRED_COMMAND:
        if key not in command:
            raise NativeStartupError(f'transition command lacks {key!r}')
    if engine_epoch is not None and command['engine_epoch'] != engine_epoch:
        raise NativeStartupError('transition command names another engine epoch')
    controller = native.controller
    if command['expected_route_generation'] != controller.route_generation:
        raise NativeStartupError('transition command expects another route generation')
    if command['route_generation'] != controller.route_generation + 1:
        raise NativeStartupError('transition command must advance the route generation by one')
    target = command['target']
    if target == SPLIT_BANK_NAME:
        target = SPLIT_ROUTE
    if target not in (FULL_ROUTE, SPLIT_ROUTE):
        raise NativeStartupError(f'unknown transition target {target!r}')
    storage_epoch = command.get('storage_epoch')
    lane = getattr(runner, '_flash_e3_lane', None)
    handle = getattr(getattr(getattr(lane, 'state', None), 'decode', None), 'handle', None)
    if storage_epoch is not None and handle is not None and storage_epoch != handle.storage_epoch:
        raise NativeStartupError('transition command names a stale storage epoch')
    started = clock()
    record = controller.enter_split() if target == SPLIT_ROUTE else controller.leave_split()
    router = getattr(runner, '_flash_e3_graph_router', None)
    authorization = command.get('authorization')
    if router is not None and authorization is not None:
        router.transition(SPLIT_BANK_NAME if target == SPLIT_ROUTE else 'ordinary', authorization)
    return {
        'ack': 'flash_s5_transition',
        'route': controller.route,
        'route_generation': controller.route_generation,
        'drained_streams': record.drained_streams,
        'switch_seconds': record.seconds,
        'ack_seconds': clock() - started,
        'decode_replay_watermark': command.get('decode_replay_watermark'),
        'storage_epoch': storage_epoch,
    }


__all__ = ['FULL_ROUTE', 'NativeStartupError', 'SPLIT_BANK_NAME', 'SPLIT_ROUTE', 'WorkerNative',
           'install_worker_native', 'transition_rpc']
