"""Generated-source seam and lane driver for the resumable model forward (D4/D2 boundary).

``models/qwen4_exp/nvidia/model.py`` is pinned and gets three additive methods
on ``Qwen4ExpModel`` (the stock ``forward`` is untouched byte-for-byte):

* ``_flash_e3_forward_prefix``: embedding (stage 0) or received HC tensor
  (stage 1); no PLE look-ahead;
* ``_flash_e3_forward_layer``: exactly one decoder layer. The PLE prefetch for
  that same layer is issued inline just before it (design: no cross-layer
  look-ahead in the lane), so nothing is carried between segments except the
  HC triple ``(hidden_states, block_output, injection)``;
* ``_flash_e3_layer_pre``/``_flash_e3_layer_post`` on the decoder layer: the HC/PLE
  part before the mixer and the HC/MoE part after it, so a QSA layer can pause
  inside its attention (``qsa_driver``);
* ``_flash_e3_forward_suffix``: stage 0 materializes the pending HC combine and
  returns the PP tensor. Stage 1 bulk chunks need no suffix: the lane never
  samples and the final mixer writes no state.

:class:`LaneForward` drives those methods one segment per call under the lane
scopes (prefill stream, PyNccl-only TP communicator, private PLE prefetch
buffer/stream, the lane forward context) and records one event per segment on
the prefill stream. Deepstack (vision) and MTP capture are refused in the lane.

The QSA attention inside a layer is D2's resumable operation; this driver calls
the layer and lets an installed ``attention_driver`` take over a QSA layer when
one is present, so the bucket credit stays with that operation.
"""
import ast
import hashlib
import time
from contextlib import ExitStack, contextmanager, nullcontext

from e3_types import E3ContractError

PINS = {
    'models/qwen4_exp/nvidia/model.py':
        'ae72d95610b80188df2ee54a8649329a2a5601c4c31cbc7b60bbda9c6dcff9c4',
}

_ANCHOR_OLD = '''        return sample_hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
'''

_METHODS = '''        return sample_hidden_states

    # --- Flash-Next E3 lane: resumable prefix / layer / suffix decomposition ---
    # Additive; the stock forward above is unchanged. The lane calls these one
    # segment at a time. No deepstack, no MTP buffer, no PLE look-ahead.
    def _flash_e3_forward_prefix(
        self,
        input_ids: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]
        return hidden_states

    def _flash_e3_forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor | None,
        injection: torch.Tensor | None,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not (self.start_layer <= layer_idx < self.end_layer):
            raise ValueError("layer index is not on this pipeline stage")
        layer = self.layers[layer_idx]
        # Same-segment prefetch: the look-up is issued and consumed inside this
        # one bounded host operation, never across a decode step.
        self._start_layer_ple_prefetch(
            layer,
            hidden_states,
            input_ids,
            query_start_loc,
            ngram_context,
        )
        return layer(
            hidden_states=hidden_states,
            prev_block_output=block_output,
            prev_injection=injection,
            positions=positions,
            input_ids=input_ids,
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )

    def _flash_e3_forward_suffix(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor | None,
        injection: torch.Tensor | None,
        last_layer_idx: int,
    ) -> IntermediateTensors:
        if get_pp_group().is_last_rank:
            raise ValueError("the lane has no last-stage suffix: bulk chunks are not sampled")
        if block_output is not None:
            hidden_states = self.layers[last_layer_idx].mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
        return IntermediateTensors({"hidden_states": hidden_states})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
'''


_LAYER_ANCHOR_OLD = '''        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection
'''

_LAYER_METHODS = '''        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection

    # --- Flash-Next E3 lane: the HC/PLE part before the mixer and the HC/MoE part after it.
    # Byte-copied from forward above so a QSA layer can pause inside its attention.
    def _flash_e3_layer_pre(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | None,
        prev_injection: torch.Tensor | None,
        *,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prev_block_output is None:
            assert prev_injection is None
        attn_hc = self.attn_hyper_connection
        if self.ple is not None:
            if prev_block_output is not None:
                hidden_states = attn_hc.combine(
                    hidden_states, prev_block_output, prev_injection
                )
                prev_block_output = prev_injection = None

            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            hidden_states = hidden_states + self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )

        if prev_block_output is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection
            )
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)
        return hidden_states, block_input, injection

    def _flash_e3_layer_post(
        self,
        hidden_states: torch.Tensor,
        attn_out: torch.Tensor,
        injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(
            hidden_states, attn_out, injection
        )
        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection
'''


class ModelHookError(E3ContractError):
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
        raise ValueError('model hooks do not transform ' + path)
    if '_flash_e3_forward_' in source:
        raise ValueError('source already carries the lane forward methods')
    if _sha(source) != pinned:
        raise ValueError('source is not the pinned bytes: ' + path)
    text = _replace(source, _ANCHOR_OLD, _METHODS)
    text = _replace(text, _LAYER_ANCHOR_OLD, _LAYER_METHODS)
    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise ValueError('generated model source syntax invalid') from exc
    return text


def revert_one(path, source, expected=None):
    if path not in PINS:
        raise ValueError('model hooks do not transform ' + path)
    if '_flash_e3_forward_' not in source:
        raise ValueError('source does not carry the lane forward methods')
    text = _replace(source, _LAYER_METHODS, _LAYER_ANCHOR_OLD)
    text = _replace(text, _METHODS, _ANCHOR_OLD)
    if expected is not None and text != expected:
        raise ValueError('revert did not recover the supplied source')
    if _sha(text) != PINS[path]:
        raise ValueError('revert did not recover the pinned bytes')
    return text


def generate(files):
    out = {}
    for path in PINS:
        if path not in files:
            raise ValueError('missing pinned input ' + path)
        out[path] = generate_one(path, files[path])
    return out


# --- runtime driver ------------------------------------------------------------

class AttributeSwap:
    """Bind private attributes on an owner for one scope; always restored."""

    def __init__(self, owner, values):
        self.owner = owner
        self.values = dict(values)
        self._saved = None

    def __enter__(self):
        if self._saved is not None:
            raise ModelHookError('attribute swap is not reentrant')
        self._saved = {}
        for name, value in self.values.items():
            if not hasattr(self.owner, name):
                raise ModelHookError(f'owner has no attribute {name!r} to bind')
            self._saved[name] = getattr(self.owner, name)
            setattr(self.owner, name, value)
        return self

    def __exit__(self, *exc):
        saved, self._saved = self._saved, None
        for name, value in saved.items():
            setattr(self.owner, name, value)
        return False


def ple_embedding(model, layer_idx):
    """The pinned-table PLE embedding of a layer, or None when it has none."""
    layer = model.layers[layer_idx]
    ple = getattr(layer, 'ple', None)
    if ple is None:
        return None
    embedding = getattr(ple, 'ngram_embedding', None)
    if embedding is None or not getattr(embedding, 'supports_prefetch', False):
        return None
    return embedding


class PlePrivateScratch:
    """Lane-private prefetch stream/buffer for the one PLE layer."""

    def __init__(self, embedding, chunk_tokens, *, torch_module, device):
        buffer = getattr(embedding, '_prefetch_buffer', None)
        stream = getattr(embedding, '_prefetch_stream', None)
        if buffer is None or stream is None:
            raise ModelHookError('PLE embedding has no prefetch buffer/stream to privatize')
        rows = int(chunk_tokens) * int(getattr(embedding, 'etp_data_parallel_size', 1))
        shape = (rows,) + tuple(buffer.shape[1:])
        self.buffer = torch_module.empty(shape, dtype=buffer.dtype, device=device)
        self.stream = torch_module.cuda.Stream(device=device)
        self.bytes = int(self.buffer.numel() * self.buffer.element_size())
        self.embedding = embedding

    def scope(self):
        return AttributeSwap(self.embedding, {'_prefetch_buffer': self.buffer,
                                              '_prefetch_stream': self.stream})


class SegmentRecord:
    __slots__ = ('layer_idx', 'event', 'host_ns', 'kind')

    def __init__(self, layer_idx, event, host_ns, kind):
        self.layer_idx = layer_idx
        self.event = event
        self.host_ns = host_ns
        self.kind = kind


class LaneForward:
    """One chunk's resumable forward on this rank, one segment per call."""

    def __init__(self, model, inputs, *, stream, tp_group=None, tp_comm=None,
                 forward_context=None, ple_scratch=None, attention_driver=None,
                 torch_module=None, clock=time.monotonic_ns):
        if not all(callable(getattr(model, m, None)) for m in (
                '_flash_e3_forward_prefix', '_flash_e3_forward_layer', '_flash_e3_forward_suffix')):
            raise ModelHookError('model does not carry the generated lane forward methods')
        for name in ('positions', 'input_ids', 'query_start_loc', 'ngram_context'):
            if name not in inputs:
                raise ModelHookError(f'lane inputs are missing {name!r}')
        if inputs.get('deepstack_input_embeds') is not None:
            raise ModelHookError('deepstack inputs are not supported in the lane')
        if getattr(model, '_mtp_hidden_buffer', None) is not None:
            raise ModelHookError('MTP hidden buffer is not supported in the lane')
        if (tp_group is None) != (tp_comm is None):
            raise ModelHookError('TP group and lane communicator go together')
        self.model = model
        self.inputs = dict(inputs)
        self.stream = stream
        self.tp_group = tp_group
        self.tp_comm = tp_comm
        self.forward_context = forward_context
        self.ple_scratch = ple_scratch
        self.attention_driver = attention_driver
        self.torch = torch_module
        self.clock = clock
        self.start_layer = int(model.start_layer)
        self.end_layer = int(model.end_layer)
        self.cursor = None
        self.hidden_states = None
        self.block_output = None
        self.injection = None
        self.segments = []
        self.state = 'new'
        self.output = None

    # -- scopes -------------------------------------------------------------
    @contextmanager
    def _scopes(self, layer_idx=None):
        torch = self.torch
        with ExitStack() as stack:
            if torch is not None:
                stack.enter_context(torch.cuda.stream(self.stream))
            if self.forward_context is not None:
                stack.enter_context(self.forward_context())
            if self.tp_group is not None:
                from lane_comm import tp_scope

                stack.enter_context(tp_scope(self.tp_group, self.tp_comm))
            if (layer_idx is not None and self.ple_scratch is not None
                    and ple_embedding(self.model, layer_idx) is self.ple_scratch.embedding):
                stack.enter_context(self.ple_scratch.scope())
            else:
                stack.enter_context(nullcontext())
            yield

    def _record(self, layer_idx, kind, started):
        event = None
        if self.torch is not None:
            event = self.torch.cuda.Event(enable_timing=False)
            event.record(self.stream)
        record = SegmentRecord(layer_idx, event, self.clock() - started, kind)
        self.segments.append(record)
        return record

    # -- segments -----------------------------------------------------------
    def begin(self, intermediate_tensors=None, inputs_embeds=None):
        if self.state != 'new':
            raise ModelHookError(f'lane forward cannot begin from state {self.state!r}')
        started = self.clock()
        with self._scopes():
            self.hidden_states = self.model._flash_e3_forward_prefix(
                self.inputs['input_ids'], intermediate_tensors, inputs_embeds)
        self.cursor = self.start_layer
        self.state = 'running'
        return self._record(None, 'prefix', started)

    def layer(self, layer_idx):
        if self.state != 'running':
            raise ModelHookError('lane forward is not running')
        if layer_idx != self.cursor:
            raise ModelHookError(f'segment {layer_idx} is out of order; cursor is {self.cursor}')
        if not (self.start_layer <= layer_idx < self.end_layer):
            raise ModelHookError('layer index is not on this stage')
        started = self.clock()
        with self._scopes(layer_idx):
            driver = self.attention_driver
            if driver is not None and driver.owns(self.model, layer_idx):
                result = driver.run_layer(self, layer_idx)
            else:
                result = self.model._flash_e3_forward_layer(
                    layer_idx, self.hidden_states, self.block_output, self.injection,
                    self.inputs['positions'], self.inputs['input_ids'],
                    self.inputs['query_start_loc'], self.inputs['ngram_context'])
        if result is None:
            # The attention driver paused inside the layer (bucket credit);
            # the cursor stays and the next call resumes the same layer.
            return self._record(layer_idx, 'layer-paused', started)
        self.hidden_states, self.block_output, self.injection = result
        self.cursor = layer_idx + 1
        return self._record(layer_idx, 'layer', started)

    def finish(self):
        if self.state != 'running':
            raise ModelHookError('lane forward is not running')
        if self.cursor != self.end_layer:
            raise ModelHookError(f'layers {self.cursor}..{self.end_layer - 1} are not done')
        started = self.clock()
        with self._scopes():
            if not self.model_is_last_stage():
                self.output = self.model._flash_e3_forward_suffix(
                    self.hidden_states, self.block_output, self.injection, self.end_layer - 1)
            else:
                self.output = None
        self.state = 'finished'
        return self._record(None, 'suffix', started)

    def model_is_last_stage(self):
        probe = getattr(self.model, '_flash_e3_is_last_stage', None)
        if probe is not None:
            return bool(probe())
        from vllm.distributed.parallel_state import get_pp_group

        return bool(get_pp_group().is_last_rank)

    def abort(self, reason='aborted'):
        self.state = 'aborted'
        self.output = None
        return reason

    def remaining(self):
        if self.cursor is None:
            return tuple(range(self.start_layer, self.end_layer))
        return tuple(range(self.cursor, self.end_layer))


__all__ = ['AttributeSwap', 'LaneForward', 'ModelHookError', 'PINS', 'PlePrivateScratch',
           'SegmentRecord', 'generate', 'generate_one', 'ple_embedding', 'revert_one']
