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
import os
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
# below; native15 adds only helper-entry shape observation to that intermediate
# source. The E2 source pin, attention methods and every stock byte are unchanged.
PINS = {
    'models/qwen4_exp/nvidia/qsa.py':
        'a0109b900e1389b2cdaa47ffb030f92b8d58d722119da69356069fcc71458a5d',
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
                 'started_ns', 'advances', 'selections', 'boundary', 'grant')

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
        self.boundary = None
        self.grant = None


def _note_indexer_scores(operation, forward_context, attn, num_tokens):
    """M3 receipt: the sizes of the indexer's FP32 score allocations for this layer.

    Reads the same host-known values the reference's qsa_select_paged_prefill
    sizes from.  No device work.  A failure is counted, never raised.
    """
    try:
        indexer = attn.indexer
        if getattr(indexer, 'skip_topk', False):
            return
        metadata = forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        compressed = metadata[indexer.compressed_key_cache.prefix]
        rows = num_tokens - compressed.num_decode_tokens
        if rows <= 0:
            return
        operation.note_indexer_scores(
            rows, compressed.max_seq_len, indexer.compress_ratio,
            compressed.block_table.shape[1], indexer.compressed_key_cache.kv_cache.shape[1],
            int(os.environ.get('VLLM_SPARSE_INDEXER_MAX_LOGITS_MB', '512')))
    except Exception:
        operation.stats['indexer_errors'] += 1


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
        self._fused_grant = None

    # -- LaneForward protocol ------------------------------------------------------
    def owns(self, model, layer_idx):
        layer = model.layers[layer_idx]
        if getattr(layer, 'layer_type', None) != 'full_attention':
            return False
        attn = getattr(layer, 'self_attn', None)
        return attn is not None and hasattr(attn, '_flash_e3_attention_pre')

    def run_layer(self, lane, layer_idx):
        phase = self.phases.get(layer_idx)
        if self._fused_grant is not None:
            return self._fused(lane, layer_idx)
        if phase is None:
            return self._pre(lane, layer_idx)
        if phase.phase == 'bucket':
            return self._bucket(lane, phase)
        if phase.phase == 'tail-ready':
            # The previous accepted layer command exposed a typed tail
            # boundary. The engine starts this command only after both active
            # TP peers accepted that boundary, so neither peer enters o_proj
            # while the other still waits for its final consumer event.
            return self._tail(lane, phase)
        raise QsaDriverError(f'layer {layer_idx} is in unexpected phase {phase.phase!r}')

    def submission_credit(self, layer_idx):
        """Host-query-only credit for the next call on ``layer_idx``.

        ``None`` means this driver has no credit seam for that layer: no phase
        is outstanding yet (the next call is the real ``pre`` submission), or
        the operation predates the seam.  The dispatcher must then keep the old
        paused behaviour.  A credit object is always the operation's own
        queried answer, never a guessed elapsed time or a cached flag.
        """
        phase = self.phases.get(layer_idx)
        if phase is None or phase.phase != 'bucket':
            return None
        query = getattr(self.operation, 'submission_credit', None)
        if not callable(query):
            return None
        return query()

    def at_schedule_boundary(self, layer_idx):
        phase = self.phases.get(layer_idx)
        return phase is not None and phase.boundary is not None and phase.grant is None

    def at_tail_boundary(self, layer_idx):
        phase = self.phases.get(layer_idx)
        return phase is not None and phase.phase == 'tail-ready'

    def schedule_profile(self, query, rank):
        from wire_v1 import QsaProfileObservation
        started = time.monotonic_ns()
        phase = self.phases.get(query.layer)
        if phase is None or phase.boundary is None or phase.grant is not None:
            raise QsaDriverError('QSA profile query is outside its submitted boundary')
        shape = self.operation.schedule_profile()
        if shape is None:
            return QsaProfileObservation(query, rank, 'pending',
                worker_host_ns=max(0, time.monotonic_ns() - started))
        run = self.operation._current_run()
        basis = ('queried-selection-copy' if run.sparse is not None
                 and run.sparse.selection is not None else 'bound-dense-host-plan')
        selected = (run.sparse.selection_count if run.sparse is not None
                    and run.sparse.selection is not None else run.valid_tokens)
        return QsaProfileObservation(query, rank, 'ready', shape, basis,
            selected_count=selected, worker_host_ns=max(0, time.monotonic_ns() - started))

    def authorize_profile(self, layer_idx, grant):
        from wire_v1 import QsaProfileGrant
        phase = self.phases.get(layer_idx)
        if phase is None or phase.boundary is None:
            if grant is not None:
                raise QsaDriverError('QSA grant precedes its submitted boundary')
            return
        if grant is None:
            raise E3Pending('QSA schedule awaits active-stage TP agreement')
        if type(grant) is not QsaProfileGrant or grant.query.layer != layer_idx:
            raise QsaDriverError('QSA grant belongs to another layer')
        if phase.grant is not None and phase.grant != grant:
            raise QsaDriverError('QSA grant changed within the layer')
        self.operation.authorize_schedule(grant)
        phase.grant = grant

    # -- Step A: one command per dense QSA layer ----------------------------------
    def fused_profiles(self, lane, valid_tokens):
        """This stage's dense schedules, agreed at PREPARE before any collective."""
        reason = self.operation.fused_refusal()
        if reason is not None:
            raise QsaDriverError('fused QSA refused at PREPARE: ' + reason)
        rows = []
        for index in range(lane.start_layer, lane.end_layer):
            if self.owns(lane.model, index):
                attn = lane.model.layers[index].self_attn
                rows.append((index, self.operation.fused_shape(
                    attn.layer_name, valid_tokens, attn.kv_cache.shape[2])))
        return tuple(rows)

    def arm_fused(self, layer_idx, grant):
        """The next run_layer on ``layer_idx`` is the whole layer. Enqueues nothing."""
        if self.phases or self._fused_grant is not None:
            raise QsaDriverError('a fused layer needs no QSA layer in progress')
        if grant.query.layer != layer_idx or grant.fused is not True:
            raise QsaDriverError('fused grant belongs to another layer')
        self._fused_grant = grant

    def _fused(self, lane, layer_idx):
        """pre, every bucket and tail in one call; the cursor advances, no pause.

        There is no pending return here: a peer can never accept this command
        while the other refuses it for credit. Any failure after pre is partial
        submission, which the dispatcher quarantines.
        """
        grant, self._fused_grant = self._fused_grant, None
        if grant.query.layer != layer_idx:
            raise QsaDriverError('fused grant belongs to another layer')
        phases = getattr(self.operation, 'phase_trace', None)
        if phases is not None and len(phases.fused_layers) < phases.MAX_LAYERS:
            phases.fused_layers.add(lane.model.layers[layer_idx].self_attn.layer_name)
        self._pre(lane, layer_idx)
        phase = self.phases[layer_idx]
        operation = self.operation
        while True:
            try:
                value = operation.advance_fused()
            except StopIteration as stop:
                self.records.append(('attended', layer_idx, stop.value))
                break
            except (SparsePlanRefused, SparseFallbackRequired) as exc:
                self.records.append(('sparse_refusal', layer_idx, exc.refusal.reason,
                                     exc.refusal.rows, exc.refusal.nbytes))
                self.abort('sparse_refusal')
                raise QsaDriverError(
                    'a fused layer refused its dense route: ' + exc.refusal.reason) from exc
            if getattr(value, 'qsa_schedule_boundary', False):
                run = operation._current_run()
                if value is not run.schedule_boundary or phase.boundary is not None:
                    raise QsaDriverError('foreign or repeated QSA schedule boundary')
                phase.boundary = value
                operation.authorize_schedule(grant)
                phase.grant = grant
                self.records.append(('schedule_boundary', layer_idx))
            elif (getattr(value, 'sparse_selection_pending', False)
                    or getattr(value, 'sparse_copy_pending', False)):
                raise QsaDriverError('a fused layer reached a data-dependent sparse wait')
            else:
                phase.buckets += 1
                phase.advances += 1
                self.records.append(('bucket', layer_idx, phase.buckets))
        if phase.grant is None:
            raise QsaDriverError('a fused layer ended without its schedule boundary')
        phase.phase = 'tail-ready'
        return self._tail(lane, phase)

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
        from lane_chunk import score_under_cap
        from native_measurement import indexer_phase_scope
        with indexer_phase_scope(getattr(operation, 'phase_trace', None)):
            selected = score_under_cap(lambda: attn.indexer(state['projected_qk'], inputs['positions'],
                                                            attn.topk_indices_buffer[:num_tokens]))
        if tuple(selected.shape) != (num_tokens, attn.indexer.packed_output_width):
            raise QsaDriverError('QSA indexer returned an invalid selection shape')
        if getattr(operation, 'memory_receipts', False):
            _note_indexer_scores(operation, self.forward_context, attn, num_tokens)
        from lane_comm import canonical_selection
        canonical_selection(selected)
        operation.boundary_evidence(selected, selection=True)
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
        credit = self.submission_credit(phase.layer_idx)
        if credit is not None and credit.can_submit is not True:
            # No submission was made: the same command must be retried with its
            # unchanged op_seq and no new accepted operation may be recorded.
            raise E3Pending('lane submission credit pending: ' + credit.reason)
        try:
            value = operation.advance()
        except E3Pending:
            if credit is not None:
                # The seam above proved credit was available and ``advance``
                # checks the same monotonic predicates before pulling the
                # continuation, so this is not a pre-submission credit wait.
                # Never swallow it: an unknown frontier stays pending/partial
                # for the dispatcher to quarantine.
                raise
            self.records.append(('pending', phase.layer_idx))
            return None
        except StopIteration as stop:
            span = stop.value
            self.records.append(('attended', phase.layer_idx, span))
            # Do not call the potentially blocking TP o_proj collective from
            # the same command that first observes this rank's final event.
            # A paused typed acceptance lets the engine wait for its peer's
            # corresponding boundary before issuing the next all-rank layer
            # command. Stop before that command cancels both without a tail.
            phase.phase = 'tail-ready'
            return None
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
        if getattr(value, 'qsa_schedule_boundary', False):
            run = operation._current_run()
            if value is not run.schedule_boundary or phase.boundary is not None:
                raise QsaDriverError('foreign or repeated QSA schedule boundary')
            phase.boundary = value
            phase.selections += 1
            self.records.append(('schedule_boundary', phase.layer_idx))
            return None
        if getattr(value, 'sparse_selection_pending', False):
            phase.selections += 1
            self.records.append(('selection', phase.layer_idx))
            return None
        if getattr(value, 'sparse_copy_pending', False):
            self.records.append(('copy_wave', phase.layer_idx))
            return None
        phase.buckets += 1
        phase.advances += 1
        self.records.append(('bucket', phase.layer_idx, phase.buckets))
        return None

    def _tail(self, lane, phase):
        layer = lane.model.layers[phase.layer_idx]
        attn = layer.self_attn
        self.operation.boundary_evidence(phase.attn_state['attn_output'])
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
