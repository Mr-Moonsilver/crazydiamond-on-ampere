"""Generated-source seams for E3 fixed-split graph routing (B).

Composition order is ``pinned -> E2 (optional) -> A lane_hooks -> B graph_hooks``.
The accepted inputs are exactly A's three reviewed full-file compositions; a
pinned or E2-only file is refused because A must install its lane identities
first.  Every transform has one unique anchor, is reversible byte-for-byte and
changes no stock code path when no graph router is installed.

The seams are deliberately narrow:

``check_route``      before the zero-token early return, queries the route and
                     refuses an impossible overlap route without waiting;
``replay_fullgraph`` wraps only the existing FULL replay call (ordinary mode
                     still executes the stock manager method unchanged);
``capture_banks``    after the stock graph capture, captures the declared split
                     bank through C's backend;
``shutdown_banks``   inside the stock shutdown, after the device synchronize
                     that C's lifecycle requires and before the stock graph
                     manager is dropped.

B never installs a forward context, workspace, stream or dispatch owner.  C
supplies the backend, the context/stream probe and the operation tickets.
"""
import ast
import hashlib
import os

from e3_types import OperationTicket, TransactionId
from graph_router import DecodeProbeRouter, ExecutionDescriptor, GraphRouter, GraphRoutingError

PINNED_SHA256 = 'a49588ad1ab173b1b5b1695d28fa64da7bbfc5a13d7d32d7aaaf9781879069a2'
PROBE_ENV = 'FLASH_S5_DECODE_ONLY_PROBE'

# Reviewed A-generated runner compositions (pinned, E2 execution, E2 startup).
A_SOURCE_SHA256 = {
    'e3': '761e67eaa8a107739846810307f2b7a1f77d3775036a20ede19c6c27be82e0c7',
    'e3-e2-exec': '40b24aee9e446d188cabe5e88aedea1efe7813960231d019ab46fbb245a87064',
    'e3-e2-startup': '08c0efde0b7a133b553ad2a40d44c44fec44066d0cc53d3fc3509ebeb7335918',
}

_IMPORT_OLD = 'import vllm.v1.flash_e3.lane_hooks as _flash_e3_hooks\n'
_IMPORT_NEW = _IMPORT_OLD + '\nimport vllm.v1.flash_e3.graph_hooks as _flash_e3_graph\n'

_CHECK_OLD = '        if batch_desc.num_tokens == 0:\n'
_CHECK_NEW = ('        _flash_e3_graph.check_route(\n'
              '            self, scheduler_output, batch_desc,\n'
              '            num_reqs, num_toks, uniform_tok_count, batch_req_state,\n'
              '        )\n' + _CHECK_OLD)

_EARLY_OLD = '            if scheduler_output.total_num_scheduled_tokens == 0:\n'
_EARLY_NEW = ('            _flash_e3_graph.check_empty(self, scheduler_output)\n' + _EARLY_OLD)

_REPLAY_OLD = '            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)\n'
_REPLAY_NEW = ('            model_output = _flash_e3_graph.replay_fullgraph(\n'
               '                self, scheduler_output, batch_desc,\n'
               '                self.cudagraph_manager.run_fullgraph,\n'
               '            )\n')

_CAPTURE_OLD = '''                        lora_capture_hook=create_lora_capture_hook(
                            self.lora_config, self
                        ),
                    )
'''
_CAPTURE_NEW = _CAPTURE_OLD + '                    _flash_e3_graph.capture_banks(self, profile_only=profile_only)\n'

_SHUTDOWN_PLAIN_OLD = ('        torch.accelerator.synchronize()\n'
                       '        self.cudagraph_manager = None\n')
_SHUTDOWN_PLAIN_NEW = ('        torch.accelerator.synchronize()\n'
                       '        _flash_e3_graph.shutdown_banks(self)\n'
                       '        self.cudagraph_manager = None\n')

_SHUTDOWN_E2_OLD = ('        torch.accelerator.synchronize()\n'
                    '        _e2 = getattr(self, "_flash_e2_lifecycle", None)\n')
_SHUTDOWN_E2_NEW = ('        torch.accelerator.synchronize()\n'
                    '        _flash_e3_graph.shutdown_banks(self)\n'
                    '        _e2 = getattr(self, "_flash_e2_lifecycle", None)\n')


def _replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('source anchor changed or duplicated: ' +
                         old[:80].replace('\n', '\\n'))
    return text.replace(old, new, 1)


def source_kind(source):
    """Return the A composition kind; reject pinned/drifted/double-applied."""
    if '_flash_e3_graph' in source:
        raise ValueError('source already carries E3 graph hooks')
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest == PINNED_SHA256:
        raise ValueError('E3 graph routing composes after E3 lane hooks')
    for kind, pinned in A_SOURCE_SHA256.items():
        if digest == pinned:
            return kind
    raise ValueError('unreviewed E3 lane source: ' + digest)


def _forward(source):
    """Ordered B transforms on top of an A-composed runner."""
    text = _replace(source, _IMPORT_OLD, _IMPORT_NEW)
    text = _replace(text, _CHECK_OLD, _CHECK_NEW)
    text = _replace(text, _EARLY_OLD, _EARLY_NEW)
    text = _replace(text, _REPLAY_OLD, _REPLAY_NEW)
    text = _replace(text, _CAPTURE_OLD, _CAPTURE_NEW)
    if '_flash_e2_lifecycle' in source:
        text = _replace(text, _SHUTDOWN_E2_OLD, _SHUTDOWN_E2_NEW)
    else:
        text = _replace(text, _SHUTDOWN_PLAIN_OLD, _SHUTDOWN_PLAIN_NEW)
    return text


def generate(source):
    """Return the B-transformed runner source on top of A's review pins."""
    source_kind(source)
    text = _forward(source)
    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise ValueError('generated runner syntax invalid') from exc
    return text


def revert(source, expected=None):
    """Exact inverse; the result must be one of A's reviewed compositions."""
    if '_flash_e3_graph' not in source:
        raise ValueError('source does not carry E3 graph hooks')
    text = source
    if '_flash_e2_lifecycle' in text:
        text = _replace(text, _SHUTDOWN_E2_NEW, _SHUTDOWN_E2_OLD)
    else:
        text = _replace(text, _SHUTDOWN_PLAIN_NEW, _SHUTDOWN_PLAIN_OLD)
    text = _replace(text, _CAPTURE_NEW, _CAPTURE_OLD)
    text = _replace(text, _REPLAY_NEW, _REPLAY_OLD)
    text = _replace(text, _EARLY_NEW, _EARLY_OLD)
    text = _replace(text, _CHECK_NEW, _CHECK_OLD)
    text = _replace(text, _IMPORT_NEW, _IMPORT_OLD)
    if expected is not None:
        if text != expected:
            raise ValueError('revert did not recover the supplied source')
    else:
        source_kind(text)
    return text


def _overlap_enabled():
    return os.environ.get('FLASH_S5_OVERLAP', '0') == '1'


def _probe_enabled():
    return os.environ.get(PROBE_ENV, '0') == '1'


def _router(runner):
    router = getattr(runner, '_flash_e3_graph_router', None)
    if router is not None and not isinstance(router, (GraphRouter, DecodeProbeRouter)):
        raise GraphRoutingError('runner E3 graph router is not a GraphRouter')
    return router


def _probe(runner):
    """The decode-only probe, when this runner started in that diagnostic mode."""
    probe = getattr(runner, '_flash_s5_decode_probe', None)
    if probe is None:
        return None
    from decode_probe import DecodeProbe

    if not isinstance(probe, DecodeProbe):
        raise GraphRoutingError('runner decode probe is not a DecodeProbe')
    if not isinstance(getattr(runner, '_flash_e3_graph_router', None), DecodeProbeRouter):
        raise GraphRoutingError('decode probe runner has no split-only router')
    return probe


def _transaction(scheduler_output, router):
    if isinstance(router, DecodeProbeRouter):
        if getattr(scheduler_output, 'flash_e3_transaction', None) is not None:
            raise GraphRoutingError('decode-only probe has no prefill transaction')
        # Typed no-prefill guard in place of the anchor transaction.
        return router.no_prefill_guard()
    transaction = getattr(scheduler_output, 'flash_e3_transaction', None)
    if transaction is None and (router.worker.active_prefill is not None
                                or router.current_mode != 'ordinary'):
        raise GraphRoutingError('split/active prefill requires an explicit decode transaction')
    return transaction


def _descriptor(batch_desc, num_reqs, num_tokens, uniform_tokens):
    for name, value in (('num_reqs', num_reqs), ('num_tokens', num_tokens)):
        if type(value) is not int or value < 0:
            raise GraphRoutingError(f'dispatched {name} is not a non-negative int')
    if uniform_tokens is not None and (type(uniform_tokens) is not int or uniform_tokens < 0):
        raise GraphRoutingError('dispatched uniform token count is invalid')
    return ExecutionDescriptor.from_vllm(batch_desc)


def _is_prefill(router, transaction):
    active = router.worker.active_prefill
    return active is not None and transaction == active


def check_empty(runner, output):
    probe = _probe(runner)
    if probe is not None and output.total_num_scheduled_tokens == 0:
        probe.observe_empty(output)
        if probe.route == 'split':
            probe.invalidate('empty_batch')
            from decode_probe import DecodeProbeInvalidated
            raise DecodeProbeInvalidated('empty batch invalidates split interval')


def check_route(runner, scheduler_output, batch_desc,
                num_reqs, num_tokens, uniform_tokens, batch_state=None):
    """Query the route before the engine commits to a graph dispatch."""
    router = _router(runner)
    if router is None:
        return None
    if batch_desc is None:
        raise GraphRoutingError("router dispatch requires a descriptor")
    if isinstance(router, DecodeProbeRouter):
        probe = _probe(runner)
        if probe is None:
            raise GraphRoutingError('decode probe router installed without the probe')
        probe.check_batch(num_reqs, num_tokens, uniform_tokens, batch_state)
        descriptor = _descriptor(batch_desc, num_reqs, num_tokens, uniform_tokens)
        return probe.check_route(descriptor, _transaction(scheduler_output, router))
    transaction = _transaction(scheduler_output, router)
    if transaction is None:
        return None
    if type(transaction) is not TransactionId:
        raise GraphRoutingError('scheduler output carries an invalid E3 transaction')
    descriptor = _descriptor(batch_desc, num_reqs, num_tokens, uniform_tokens)
    if descriptor.num_tokens == 0:
        # The stock engine returns before dispatch; no graph can be replayed.
        return None
    if _is_prefill(router, transaction):
        # A prefill never replays a decode FULL graph and never touches the
        # decode sampling ring; piecewise/eager prefill still runs stock.
        if descriptor.cg_mode == 'full':
            raise GraphRoutingError('prefill cannot replay a decode FULL graph')
        return None
    decision = router.supported_route(transaction, descriptor)
    if decision.mode == 'split' and descriptor.num_tokens > 0:
        if type(getattr(scheduler_output, 'flash_e3_decode_ticket', None)) is not OperationTicket:
            raise GraphRoutingError('split route needs C\'s registered decode ticket')
    return decision


def replay_fullgraph(runner, scheduler_output, batch_desc, stock):
    """Replay the split bank, or run exactly the stock FULL call."""
    router = _router(runner)
    if router is None:
        return stock(batch_desc)
    if batch_desc is None:
        raise GraphRoutingError("router replay requires a descriptor")
    if isinstance(router, DecodeProbeRouter):
        probe = _probe(runner)
        if probe is None:
            raise GraphRoutingError('decode probe router installed without the probe')
        probe._require_not_down()
        if probe.route != 'split':
            return stock(batch_desc)
        return probe.replay(ExecutionDescriptor.from_vllm(batch_desc))
    transaction = _transaction(scheduler_output, router)
    if transaction is None:
        return stock(batch_desc)
    if type(transaction) is not TransactionId:
        raise GraphRoutingError('scheduler output carries an invalid E3 transaction')
    descriptor = ExecutionDescriptor.from_vllm(batch_desc)
    if _is_prefill(router, transaction):
        if descriptor.cg_mode == 'full':
            raise GraphRoutingError('prefill cannot replay a decode FULL graph')
        return stock(batch_desc)
    decision = router.supported_route(transaction, descriptor)
    if decision.mode == 'ordinary':
        return stock(batch_desc)
    ticket = getattr(scheduler_output, 'flash_e3_decode_ticket', None)
    if type(ticket) is not OperationTicket:
        raise GraphRoutingError('split replay needs C\'s registered decode ticket')
    probe = getattr(router, 'probe', None)
    if not callable(probe):
        raise GraphRoutingError('split replay needs C\'s context/stream probe')
    observed = probe()
    if (type(observed) is not tuple or len(observed) != 2
            or type(observed[0]) is not int or type(observed[1]) is not int):
        raise GraphRoutingError('C context/stream probe returned an invalid pair')
    result = router.replay(transaction, descriptor, ticket,
                           observed_context=observed[0],
                           observed_stream=observed[1])
    return result.output


def capture_banks(runner, *, profile_only=False):
    """Capture the declared split bank after the ordinary engine capture."""
    if profile_only and _probe_enabled():
        return None  # ephemeral profiling capture is not the measured bank
    router = _router(runner)
    if router is None:
        if _probe_enabled():
            from native_startup import install_decode_probe

            probe = install_decode_probe(runner)
            if getattr(runner, '_flash_e3_graph_captured', False):
                raise GraphRoutingError('graph banks were already captured')
            runner._flash_e3_graph_captured = True
            return probe.router
        if not _overlap_enabled():
            return None
        builder = getattr(runner, '_flash_e3_graph_builder', None)
        if builder is None:
            raise GraphRoutingError('FLASH_S5_OVERLAP=1 without C graph builder')
        router = builder(runner)
        if not isinstance(router, GraphRouter):
            raise GraphRoutingError('C graph builder did not return a GraphRouter')
        runner._flash_e3_graph_router = router
    if isinstance(router, DecodeProbeRouter):
        probe = _probe(runner)
        if probe is None:
            raise GraphRoutingError('decode probe router installed without the probe')
        if getattr(runner, '_flash_e3_graph_captured', False):
            raise GraphRoutingError('graph banks were already captured')
        if probe.router.state == 'empty':
            probe.capture()
        elif probe.router.state != 'ready':
            raise GraphRoutingError('decode probe bank is not capturable')
        runner._flash_e3_graph_captured = True
        return router
    if getattr(runner, '_flash_e3_graph_captured', False):
        raise GraphRoutingError('graph banks were already captured')
    router.capture_at_startup()
    runner._flash_e3_graph_captured = True
    return router


def shutdown_banks(runner):
    """Destroy graph banks after C has drained; idempotent once complete."""
    probe = _probe(runner)
    if probe is not None:
        if getattr(runner, '_flash_e3_graph_shutdown', False):
            return None
        record = probe.shutdown()
        runner._flash_e3_graph_shutdown = True
        return record
    router = _router(runner)
    if router is None:
        return None
    if getattr(runner, '_flash_e3_graph_shutdown', False):
        return None
    record = router.shutdown()
    runner._flash_e3_graph_shutdown = True
    return record


__all__ = [
    'A_SOURCE_SHA256', 'PINNED_SHA256', 'PROBE_ENV', 'capture_banks', 'check_route',
    'generate', 'replay_fullgraph', 'revert', 'shutdown_banks', 'source_kind',
]
