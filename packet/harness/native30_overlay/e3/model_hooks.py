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


def local_text_model(root):
    """Resolve the one local backbone through the pinned model owner paths.

    Qwen4ExpModel owns the local layers and HC text config (model.py:388).
    Qwen4ExpForCausalLM owns it as ``model`` (model.py:671), and the outer
    Qwen4ExpForConditionalGeneration owns that causal model as
    ``language_model`` (model.py:958), including language-model-only serving.
    Direct backbone callers remain supported. Never guess dimensions from an
    outer config or recursively search arbitrary ``model`` attributes.

    This only resolves ownership. Existing pinned-family/stream census,
    channel geometry, native methods and private-buffer guards remain in force.
    """
    found = {}
    for path in ((), ('model',), ('language_model', 'model')):
        candidate = root
        for name in path:
            candidate = getattr(candidate, name, None)
        layers = getattr(candidate, 'layers', None)
        if layers is None:
            continue
        start = getattr(candidate, 'start_layer', None)
        end = getattr(candidate, 'end_layer', None)
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end
                or not callable(getattr(layers, '__getitem__', None))):
            raise ModelHookError('local text model has an invalid layer interval')
        try:
            if end > len(layers):
                raise ModelHookError('local text model layer interval exceeds its storage')
        except TypeError as exc:
            raise ModelHookError('local text model layers have no bounded length') from exc
        found[id(candidate)] = candidate
    if len(found) != 1:
        raise ModelHookError('expected one local text backbone at the pinned owner paths')
    return next(iter(found.values()))


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


# ple-private (FINDING-ple-shared-prefetch.md): native and decode share one stock _prefetch_buffer, with nothing ordering their writes.
# FLASH_S5_PLE_PRIVATE builds the private buffer at startup; the rank lever selects it per arm. FLASH_S5_PLE_WITNESS installs the witness; its lever
# selects it per arm, because its host sync costs time. Both default off, and with both off nothing here runs.
PLE_PRIVATE = dict(selected=False, scopes=0)
PLE_WITNESS = dict(selected=False, native_checks=0, native_unequal=0, native_rows=0, decode_checks=0, decode_unequal=0, decode_rows=0, missing_ids=0, errors=0)


def select_ple_private(on):
    if type(on) is not bool:
        raise ModelHookError('ple-private-not-bool')
    PLE_PRIVATE['selected'] = on
    return dict(PLE_PRIVATE)


def select_ple_witness(on):
    if type(on) is not bool:
        raise ModelHookError('ple-witness-not-bool')
    PLE_WITNESS['selected'] = on
    return dict(PLE_WITNESS)


class PleWitness:
    """Repeats the stock lookup for a lane's first rows just before it reads them, and compares bytes.

    The same ids and the same pinned table give equal bytes, so an unequal row had another writer. Hooks the two undecorated stock methods the prefetch
    calls by attribute, so it also holds on graph replay, whose eager breaks call the undecorated functions. Rows are this rank's, before the TP reduce.
    A failure is counted, never raised. Counts only, no values. Its host read moves timing, so a zero doesn't clear a race.
    """

    ROWS = 8

    def __init__(self, embedding, native_stream, *, torch_module, device):
        buffer = embedding._prefetch_buffer
        self.embedding, self.native_stream, self.torch = embedding, native_stream, torch_module
        self.buffer = torch_module.empty((self.ROWS,) + tuple(buffer.shape[1:]), dtype=buffer.dtype, device=device)
        self.ids = {}
        self._stock_lookup, self._stock_reduce = embedding._lookup, embedding._reduce_etp_embeddings

    def install(self):
        self.embedding._lookup, self.embedding._reduce_etp_embeddings = self.lookup, self.reduce
        return self

    def lookup(self, input_ids, output=None):
        result = self._stock_lookup(input_ids, output=output)
        if PLE_WITNESS['selected'] and output is not None:
            self.ids[id(self.embedding._prefetch_stream)] = input_ids[:self.ROWS].clone()
        return result

    def reduce(self, embeddings):
        if PLE_WITNESS['selected']:
            self.check(embeddings)
        return self._stock_reduce(embeddings)

    def check(self, embeddings):
        stream = self.embedding._prefetch_stream
        ids = self.ids.pop(id(stream), None)
        if ids is None:
            PLE_WITNESS['missing_ids'] += 1
            return
        lane = 'native' if stream is self.native_stream else 'decode'
        try:
            rows = min(int(ids.shape[0]), int(embeddings.shape[0]))
            fresh = self._stock_lookup(ids[:rows], output=self.buffer[:rows])
            torch = self.torch
            read = embeddings[:rows].reshape(rows, -1).view(torch.uint8)
            equal = (read == fresh.reshape(rows, -1).view(torch.uint8)).all(dim=1)
            bad = rows - int(equal.sum().item())
        except Exception:  # noqa: BLE001 - counted, the arm carries on
            PLE_WITNESS['errors'] += 1
            return
        PLE_WITNESS[lane + '_checks'] += 1
        PLE_WITNESS[lane + '_unequal'] += int(bad > 0)
        PLE_WITNESS[lane + '_rows'] += bad


def ple_embedding(model, layer_idx, rows=1024, private=False):
    """The pinned-table PLE embedding of a layer, or None when it has none.

    Decode-phase dp5b: the deployed model holds it at layer.ple.ple_embedding.ngram_embedding, so at 1024 rows this finds nothing on the real
    model and native shares stock's _prefetch_buffer, as in every run so far. Above 1024 rows that buffer is too short: the real path resolves
    and the private scratch engages. 1024 stays as it was, so a 1024 rerun stays comparable, unless ``private`` (FLASH_S5_PLE_PRIVATE) asks for the real path there too.
    """
    layer = model.layers[layer_idx]
    ple = getattr(layer, 'ple', None)
    if ple is None:
        return None
    embedding = getattr(ple, 'ngram_embedding', None)
    if embedding is None and (rows > 1024 or private):
        embedding = getattr(getattr(ple, 'ple_embedding', None), 'ngram_embedding', None)
    if embedding is None or not getattr(embedding, 'supports_prefetch', False):
        return None
    return embedding


class PlePrivateScratch:
    """Lane-private prefetch stream/buffer for the one PLE layer.

    The stream is supplied by the caller.  The native lane passes the owned
    prefill-green auxiliary (see :class:`PrefillStreamOwners`); the default-off
    path and the CPU tests pass none and keep a private primary-context stream.
    ``stream_basis`` records which one was used.
    """

    def __init__(self, embedding, chunk_tokens, *, torch_module, device, stream=None, selectable=False):
        self.selectable = selectable and int(chunk_tokens) <= 1024     # above 1024 stock's buffer can't serve: always engaged, no lever
        buffer = getattr(embedding, '_prefetch_buffer', None)
        original = getattr(embedding, '_prefetch_stream', None)
        if buffer is None or original is None:
            raise ModelHookError('PLE embedding has no prefetch buffer/stream to privatize')
        if stream is None:
            self.stream = torch_module.cuda.Stream(device=device)
            self.stream_basis = 'primary-context'
        else:
            if not getattr(stream, 'cuda_stream', 0):
                raise ModelHookError('PLE prefetch stream is not a CUDA stream wrapper')
            self.stream = stream
            self.stream_basis = 'prefill-green'
        rows = int(chunk_tokens) * int(getattr(embedding, 'etp_data_parallel_size', 1))
        shape = (rows,) + tuple(buffer.shape[1:])
        self.buffer = torch_module.empty(shape, dtype=buffer.dtype, device=device)
        self.bytes = int(self.buffer.numel() * self.buffer.element_size())
        self.embedding = embedding
        self.chunk_tokens = int(chunk_tokens)

    def engaged(self, model, layer_idx):
        """On this layer's segment; a selectable scratch only when the lever says so. Counted either way: above 1024 rows the counter is the box's only sign of it."""
        if ple_embedding(model, layer_idx, self.chunk_tokens, self.selectable) is not self.embedding:
            return False
        if self.selectable and not PLE_PRIVATE['selected']:
            return False
        PLE_PRIVATE['scopes'] += 1
        return True

    def scope(self):
        return AttributeSwap(self.embedding, {'_prefetch_buffer': self.buffer,
                                              '_prefetch_stream': self.stream})


def prefill_pinned_types():
    """Exact stock types whose side streams the prefill lane may own.

    Same pinned sources as the decode capture repair, so the two ownership
    paths cannot drift apart.  Imported lazily: no vLLM import at startup and
    no torch import in this module.
    """
    from capture_streams import pinned_types
    return pinned_types()


class PrefillStreamOwners:
    """Persistent prefill side-stream ownership for one worker incarnation.

    Unlike ``capture_streams.CaptureStreams`` (decode graph capture only), this
    keeps one prefill-green auxiliary per distinct original side stream and
    swaps it in for every prefill segment, restoring the exact original in a
    finally.  Originals are never destroyed and the stock event topology is
    untouched.  Owners sharing one original stream keep sharing its
    replacement; a decoder auxiliary is never reused.

    ``prepare`` retains the split's allocator context BEFORE any tensor can be
    exposed to the new streams (VM7b): wrappers and green contexts outlive the
    allocator's block->stream records, so native destruction is refused from
    then on and only process teardown reclaims them.
    """

    ATTRS = ('_stream', '_prefetch_stream')

    def __init__(self, split, owners):
        if not all(callable(getattr(split, m, None)) for m in (
                'create_prefill_aux', 'check_prefill_stream', 'destroy_prefill_aux',
                'retain_allocator_context', 'lifetime_record')):
            raise ModelHookError('prefill stream owners need the green split extension')
        self.split = split
        self.owners = tuple(owners)
        self.originals = []
        self.replacements = []
        self.streams = []
        self.state = 'new'
        self._in_scope = False
        seen = set()
        for entry in self.owners:
            if type(entry) is not tuple or len(entry) != 2 or entry[1] not in self.ATTRS:
                raise ModelHookError('unsupported prefill stream owner')
            obj, attr = entry
            if (id(obj), attr) in seen:
                raise ModelHookError('duplicate prefill stream owner')
            seen.add((id(obj), attr))
            old = getattr(obj, attr, None)
            if old is not None and not getattr(old, 'cuda_stream', 0):
                raise ModelHookError('prefill stream owner has no stream')
            if old is None and attr == '_prefetch_stream':
                raise ModelHookError('PLE owner has no prefetch stream to own')
            self.originals.append(old)

    @classmethod
    def discover(cls, runner, split, *, pinned=None):
        """Find this stage's exact SharedExperts/PLE owners.

        Mirrors the owner census of ``capture_streams.for_runner``: exactly one
        local Qwen4Exp model, one SharedExperts per local layer and one PLE
        embedding per local PLE layer.  A stream disabled by the stock flag
        (``_stream is None``) is preserved as disabled, never re-enabled.
        """
        shared_type, ple_type = (pinned or prefill_pinned_types())
        modules = tuple(runner.model.modules())
        models = [m for m in modules if type(m).__name__ == 'Qwen4ExpModel'
                  and type(m).__module__ == 'vllm.models.qwen4_exp.nvidia.model']
        if len(models) != 1:
            raise ModelHookError('expected one local Qwen4Exp model for prefill streams')
        model = models[0]
        start, end = model.start_layer, model.end_layer
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ModelHookError('invalid local layer interval for prefill streams')
        shared = [m for m in modules if type(m) is shared_type]
        ple = [m for m in modules if type(m) is ple_type]
        expected_ple = sum(start < n <= end for n in set(model.config.ple_layer_ids))
        if len(shared) != end - start or len(ple) != expected_ple:
            raise ModelHookError('prefill side-stream owner census differs from local model')
        return cls(split, [(m, '_stream') for m in shared]
                   + [(m, '_prefetch_stream') for m in ple])

    def prepare(self):
        """Create one prefill auxiliary per distinct original stream."""
        if self.state != 'new':
            raise ModelHookError('prefill stream ownership is already prepared')
        groups = {}
        created = []
        try:
            for old in self.originals:
                if old is None:
                    # Stock shared-experts stream disabled: preserve that.
                    self.replacements.append(None)
                    continue
                key = old.cuda_stream
                if key not in groups:
                    stream = self.split.create_prefill_aux()
                    self.split.check_prefill_stream(stream)
                    created.append(stream)
                    groups[key] = stream
                self.replacements.append(groups[key])
            # VM7b: the allocator may already hold block->stream records for
            # these wrappers; retain the contexts before any tensor work.
            self.split.retain_allocator_context()
            self.streams = created
            self.state = 'ready'
        except BaseException:
            self.state = 'failed'
            self.replacements = []
            if not self.split.lifetime_record()['allocator_exposed']:
                for stream in reversed(created):
                    self.split.destroy_prefill_aux(stream)
            self.streams = []
            raise
        return self

    @contextmanager
    def scope(self):
        """Swap every owned prefill side stream in; restore exactly in finally."""
        if self.state != 'ready':
            raise ModelHookError('prefill stream ownership is not ready')
        if self._in_scope:
            raise ModelHookError('prefill stream scope is not reentrant')
        for (obj, attr), old in zip(self.owners, self.originals):
            if getattr(obj, attr, None) is not old:
                raise ModelHookError('prefill side-stream owner changed outside the lane scope')
        installed = []
        self._in_scope = True
        try:
            for (obj, attr), old, new in zip(self.owners, self.originals, self.replacements):
                if new is None:
                    continue
                setattr(obj, attr, new)
                installed.append((obj, attr, old))
            yield self
        finally:
            for obj, attr, old in reversed(installed):
                setattr(obj, attr, old)
            self._in_scope = False

    def replacement(self, owner, attr):
        """The owned replacement stream for one registered owner."""
        for (obj, name), new in zip(self.owners, self.replacements):
            if obj is owner and name == attr:
                if new is None:
                    raise ModelHookError('stream owner is disabled in the stock model')
                return new
        raise ModelHookError('owner is not registered for prefill streams')

    def drain(self):
        """Synchronize every owned prefill auxiliary; returns the count."""
        if self.state != 'ready':
            raise ModelHookError('prefill stream ownership is not ready')
        for stream in self.streams:
            stream.synchronize()
        return len(self.streams)

    def record(self):
        originals = {id(old) for old in self.originals if old is not None}
        return {'state': self.state,
                'owners': len(self.owners),
                'shared_owners': sum(a == '_stream' for _, a in self.owners),
                'ple_owners': sum(a == '_prefetch_stream' for _, a in self.owners),
                'distinct_original_streams': len(originals),
                'auxiliary_streams': len(self.streams),
                'basis': 'prefill-green',
                'original_owners_restored': all(
                    getattr(o, a, None) is old
                    for (o, a), old in zip(self.owners, self.originals)),
                'allocator_lifetime': self.split.lifetime_record()}


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
                 torch_module=None, clock=time.monotonic_ns, prefill_owners=None,
                 private_scope=None):
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
        self.prefill_owners = prefill_owners
        self.private_scope = private_scope
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
        from lane_private import score_model_boundary
        score_model_boundary()
        torch = self.torch
        with ExitStack() as stack:
            if torch is not None:
                inference = getattr(torch, 'inference_mode', None)
                if inference is not None:
                    stack.enter_context(inference())
                stack.enter_context(torch.cuda.stream(self.stream))
            if self.private_scope is not None:
                # The lane-private QSA top-k/UID bindings and the lane's own
                # host forward-context/workspace ownership for this segment.
                stack.enter_context(self.private_scope(layer_idx))
            if self.forward_context is not None:
                stack.enter_context(self.forward_context())
            if self.tp_group is not None:
                from lane_comm import tp_scope

                stack.enter_context(tp_scope(self.tp_group, self.tp_comm))
            if self.prefill_owners is not None:
                # Side streams that stock code runs on (SharedExperts and the
                # PLE prefetch stream) are owned by the prefill green context
                # for this segment and restored exactly afterwards.
                stack.enter_context(self.prefill_owners.scope())
            if (layer_idx is not None and self.ple_scratch is not None
                    and self.ple_scratch.engaged(self.model, layer_idx)):
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


__all__ = ['AttributeSwap', 'LaneForward', 'ModelHookError', 'PINS', 'PLE_PRIVATE', 'PLE_WITNESS', 'PlePrivateScratch', 'PleWitness', 'select_ple_private', 'select_ple_witness',
           'PrefillStreamOwners', 'SegmentRecord', 'generate', 'generate_one',
           'local_text_model', 'ple_embedding', 'prefill_pinned_types', 'revert_one']
