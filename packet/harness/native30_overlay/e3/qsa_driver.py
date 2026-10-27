"""QSA layer driver for the E3 lane: producer, one-bucket attention generator, tail (D4/D2 boundary).

With ``layer._flash_s5_operation`` installed, D2's generated QSA module refuses
the stock ``_run_qsa``/``forward_qsa`` path on purpose. This driver executes
exactly the same arithmetic in three bounded phases per QSA layer:

``pre``     the decoder layer's HC part (``_flash_e3_layer_pre``), the attention
            projections (``_flash_e3_attention_pre``), then the producer
            boundary: ``begin_layer`` -> indexer -> host K/V store ->
            ``producer_submitted``; the one-bucket generator
            ``_flash_s5_staged_operation`` is created and attached;
``bucket``  one ``operation.advance()`` per host call: enqueues at most one
            fill+attention bucket or raises E3Pending while the previous
            bucket's consumer is not host-queried complete; the lane returns
            to decode between calls.  A sparse layer's selection readback is
            its own resumed phase: the same call returns a
            ``sparse_selection_pending`` sentinel instead of blocking on the
            device, and a typed plan refusal parks the layer for the serial
            route;
``tail``    after StopIteration: gate, ``o_proj`` (TP all-reduce on the lane
            communicator), ``end_layer``, then the decoder layer's HC/MoE part
            (``_flash_e3_layer_post``).

The decode row cache is bypassed entirely (the lane never claims rows). The
QSA attention module gets two additive methods (``_flash_e3_attention_pre``/
``_post``) on the D2-composed source, pinned with an exact inverse.
"""
import ast
import hashlib
import time

from sparse_history_dma import (
    SparseFallbackRequired,
    SparsePlanRefused,
    SparseSourceHold,
)
from row1_oracle import Row1OracleError
from e3_types import E3ContractError, E3Pending

# D2-composed qsa.py: runner_hooks.qsa_staged(pinned) -> operation_source.generate_layer.
# The sparse-history worker added the compact-row helper and the fixed-capacity
# union prelude, so this intermediate pin moved from 98395b80... to the value
# below; the E2 source pin and every stock byte are unchanged.  Reported for
# Codex review.
PINS = {
    'models/qwen4_exp/nvidia/qsa.py':
        '9cfe8556ad1272e8d9a9c84c59a2cca4143520b35d49b7f34210d84480526d8f',
}

_ATTN_ANCHOR_OLD = '''        output, _ = self.o_proj(flat_output)
        return output
'''

_ATTN_METHODS = _ATTN_ANCHOR_OLD + '''
    # --- Flash-Next E3 lane: projections before and the output path after the
    # bounded attention operation.  Byte-copied from forward above; the lane's
    # qsa_driver runs the producer and the one-bucket generator in between.
    def _flash_e3_attention_pre(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> dict:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        return {
            "query": query,
            "key": key,
            "value": value,
            "gate": gate,
            "attn_output": attn_output,
            "projected_qk": projected_qk,
            "num_tokens": num_tokens,
        }

    def _flash_e3_attention_post(self, state: dict) -> torch.Tensor:
        flat_output = state["attn_output"].view(state["num_tokens"], -1)
        gate = state["gate"]
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output
'''


class QsaDriverError(E3ContractError):
    pass


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('source anchor changed or duplicated: ' +
                         old[:80].replace('\n', '\\n'))
    return text.replace(old, new, 1)


def generate_one(path, source):
    pinned = PINS.get(path)
    if pinned is None:
        raise ValueError('qsa driver does not transform ' + path)
    if '_flash_e3_attention_pre' in source:
        raise ValueError('source already carries the lane attention methods')
    if _sha(source) != pinned:
        raise ValueError('source is not the D2-composed QSA bytes: ' + path)
    text = _replace(source, _ATTN_ANCHOR_OLD, _ATTN_METHODS)
    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise ValueError('generated QSA source syntax invalid') from exc
    return text


def revert_one(path, source, expected=None):
    if path not in PINS:
        raise ValueError('qsa driver does not transform ' + path)
    if '_flash_e3_attention_pre' not in source:
        raise ValueError('source does not carry the lane attention methods')
    text = _replace(source, _ATTN_METHODS, _ATTN_ANCHOR_OLD)
    if expected is not None and text != expected:
        raise ValueError('revert did not recover the supplied source')
    if _sha(text) != PINS[path]:
        raise ValueError('revert did not recover the D2-composed bytes')
    return text


# --- runtime driver -----------------------------------------------------------------

class LayerPhase:
    __slots__ = ('layer_idx', 'hidden', 'injection', 'attn_state', 'buckets', 'phase',
                 'started_ns', 'advances', 'selections')

    def __init__(self, layer_idx, hidden, injection, attn_state):
        self.layer_idx = layer_idx
        self.hidden = hidden
        self.injection = injection
        self.attn_state = attn_state
        self.buckets = 0
        self.advances = 0
        self.selections = 0
        self.phase = 'bucket'
        self.started_ns = 0


def _metadata_for(forward_context, attn):
    context = forward_context()
    metadata = getattr(context, 'attn_metadata', None)
    if isinstance(metadata, list):
        metadata = metadata[0]
    if not isinstance(metadata, dict):
        raise QsaDriverError('lane forward context carries no per-layer attention metadata')
    try:
        main = metadata[attn.layer_name]
        side = metadata[attn.indexer.raw_key_cache.prefix]
    except KeyError as exc:
        raise QsaDriverError('lane metadata lacks the QSA main/side entries') from exc
    if side.num_actual_tokens != main.num_actual_tokens:
        raise QsaDriverError('QSA main and side metadata token counts disagree')
    return main, side


class QsaLayerDriver:
    """``attention_driver`` for ``LaneForward``: one bounded phase per call."""

    def __init__(self, operation, *, staged_operation, forward_context, torch_module=None,
                 clock=time.monotonic_ns, sparse_staged_operation=None,
                 source_holds=None):
        if operation is None or not callable(getattr(operation, 'advance', None)):
            raise QsaDriverError('driver needs the transaction QSA operation')
        if not callable(staged_operation):
            raise QsaDriverError('driver needs the generated staged-operation helper')
        if not callable(forward_context):
            raise QsaDriverError('driver needs the lane forward-context getter')
        if sparse_staged_operation is not None and not callable(sparse_staged_operation):
            raise QsaDriverError('sparse staged-operation helper must be callable')
        holds = dict(source_holds or {})
        for name, hold in holds.items():
            if not isinstance(name, str) or not name:
                raise QsaDriverError('sparse source hold needs a layer name')
            if type(hold) is not SparseSourceHold:
                raise QsaDriverError('sparse source hold must be an admitted hold')
        if holds and sparse_staged_operation is None:
            raise QsaDriverError('sparse source holds need the sparse staged helper')
        self.operation = operation
        self.staged_operation = staged_operation
        self.sparse_staged_operation = sparse_staged_operation
        self.source_holds = holds
        self.forward_context = forward_context
        self.torch = torch_module
        self.clock = clock
        self.phases = {}
        self.records = []

    # -- LaneForward protocol ------------------------------------------------------
    def owns(self, model, layer_idx):
        layer = model.layers[layer_idx]
        if getattr(layer, 'layer_type', None) != 'full_attention':
            return False
        attn = getattr(layer, 'self_attn', None)
        return attn is not None and hasattr(attn, '_flash_e3_attention_pre')

    def run_layer(self, lane, layer_idx):
        phase = self.phases.get(layer_idx)
        if phase is None:
            return self._pre(lane, layer_idx)
        if phase.phase == 'bucket':
            return self._bucket(lane, phase)
        raise QsaDriverError(f'layer {layer_idx} is in unexpected phase {phase.phase!r}')

    # -- phases -------------------------------------------------------------------
    def _pre(self, lane, layer_idx):
        if self.phases:
            raise QsaDriverError('another QSA layer is still in progress')
        layer = lane.model.layers[layer_idx]
        attn = layer.self_attn
        inputs = lane.inputs
        hidden, block_input, injection = layer._flash_e3_layer_pre(
            lane.hidden_states, lane.block_output, lane.injection,
            input_ids=inputs['input_ids'], query_start_loc=inputs['query_start_loc'],
            ngram_context=inputs['ngram_context'])
        state = attn._flash_e3_attention_pre(inputs['positions'], block_input)
        main, side = _metadata_for(self.forward_context, attn)
        num_tokens = main.num_actual_tokens
        if num_tokens != state['num_tokens']:
            raise QsaDriverError('lane metadata token count differs from the chunk rows')
        if getattr(attn, 'kv_cache', None) is None or attn.kv_cache.numel() == 0:
            raise QsaDriverError('QSA main K/V cache is not bound')
        operation = self.operation
        operation.begin_layer(attn.layer_name, attn._qsa_host_kv)
        selected = attn.indexer(state['projected_qk'], inputs['positions'],
                                attn.topk_indices_buffer[:num_tokens])
        if tuple(selected.shape) != (num_tokens, attn.indexer.packed_output_width):
            raise QsaDriverError('QSA indexer returned an invalid selection shape')
        impl = attn.impl
        impl.do_kv_cache_update(attn, state['key'], state['value'], attn.kv_cache,
                                main.slot_mapping)
        operation.producer_submitted(attn.layer_name, state['key'], state['value'], attn.kv_cache)
        hold = self.source_holds.get(attn.layer_name)
        if hold is not None:
            operation.attach_sparse_source(attn.layer_name, hold)
        helper = self.staged_operation
        if (self.sparse_staged_operation is not None
                and callable(getattr(operation, 'has_sparse_source', None))
                and operation.has_sparse_source(attn.layer_name)):
            helper = self.sparse_staged_operation
        attn_output = state['attn_output']
        attn_output.zero_()
        logical_indices = attn.topk_indices_buffer[:num_tokens]
        use_prefill_config = main.max_query_len > attn._max_decode_query_len
        generator = helper(operation, attn, state['query'], attn.kv_cache,
                           logical_indices, main, num_tokens,
                           use_prefill_config, attn_output)
        operation.attach_continuation(generator)
        phase = LayerPhase(layer_idx, hidden, injection, state)
        phase.started_ns = self.clock()
        self.phases[layer_idx] = phase
        self.records.append(('pre', layer_idx))
        return None

    def _bucket(self, lane, phase):
        operation = self.operation
        try:
            value = operation.advance()
        except E3Pending:
            self.records.append(('pending', phase.layer_idx))
            return None
        except StopIteration as stop:
            span = stop.value
            self.records.append(('attended', phase.layer_idx, span))
            return self._tail(lane, phase)
        except (SparsePlanRefused, SparseFallbackRequired) as exc:
            # Typed admission refusal: the arena was not filled beside decode.
            # Park this layer for the coordinated serial route; never count it
            # as an overlap bucket.
            refusal = exc.refusal
            self.records.append(('sparse_refusal', phase.layer_idx,
                                 refusal.reason, refusal.rows, refusal.nbytes))
            self.abort('sparse_refusal')
            raise QsaDriverError(
                'sparse history refused the overlap route: ' + refusal.reason) from exc
        except Row1OracleError as exc:
            # The actual arena/output bytes differ from the original UVA path.
            # Stop this transaction; the records live in the bounded store.
            self.records.append(('row1_failure', phase.layer_idx, str(exc)))
            self.abort('row1_failure')
            raise QsaDriverError('native row1 qualification failed: ' + str(exc)) from exc
        if getattr(value, 'sparse_selection_pending', False):
            phase.selections += 1
            self.records.append(('selection', phase.layer_idx))
            return None
        phase.buckets += 1
        phase.advances += 1
        self.records.append(('bucket', phase.layer_idx, phase.buckets))
        return None

    def _tail(self, lane, phase):
        layer = lane.model.layers[phase.layer_idx]
        attn = layer.self_attn
        attn_out = attn._flash_e3_attention_post(phase.attn_state)
        self.operation.end_layer(attn.layer_name)
        result = layer._flash_e3_layer_post(phase.hidden, attn_out, phase.injection)
        del self.phases[phase.layer_idx]
        self.records.append(('tail', phase.layer_idx, self.clock() - phase.started_ns))
        return result

    # -- introspection ------------------------------------------------------------
    def cursor(self):
        if not self.phases:
            return None
        phase = next(iter(self.phases.values()))
        return (phase.layer_idx, phase.phase, phase.buckets)

    def abort(self, reason='aborted'):
        self.phases.clear()
        if callable(getattr(self.operation, 'cancel', None)):
            return self.operation.cancel(reason)
        return reason


__all__ = ['PINS', 'LayerPhase', 'QsaDriverError', 'QsaLayerDriver', 'generate_one', 'revert_one']
