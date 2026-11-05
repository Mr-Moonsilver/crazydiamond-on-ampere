"""Private mutable bindings for the E3 prefill lane (package I).

The captured decode graph retains the *addresses* of the mutable QSA tensors it
was captured with.  ``QsaLayerDriver`` writes ``attn.topk_indices_buffer`` and
the QSA row cache reads ``vllm.forward_context.QSA_REQ_UID``.  A separately
allocated ``prefill_qsa_topk`` inventory entry is not a binding; without this
module a native prefill segment writes the very rows decode's graph reads.

This module installs the bindings that already-owned native storage supports:

* ``topk_indices_buffer`` of the *current* QSA layer becomes an int32 view of
  ``runtime._native_owned_buffers['prefill_qsa_topk']`` for one segment;
* ``QSA_REQ_UID`` becomes ``runtime.prefill_uid_buffer`` with row 0 populated
  from the admitted request's actual UID slot (``runner._qsa_uid_slots``);
* the segment runs inside ``LaneRuntime.dispatcher`` (host forward-context and
  the extra prefill workspace lane) on the lane stream;
* persistent Marlin lock-array owners use separate retained allocations with
  the exact original geometry; shared owners preserve their private sharing.

Every original is restored in a ``finally`` and never freed here.  Restoring a
Python attribute does not release GPU storage: the private view and buffer stay
owned by ``runtime._native_owned_buffers`` for the worker incarnation and by
the operation's consumer/cancel/drain proofs.

Evidence boundary: these are CPU-checked construction and refusal rules.  No
CUDA execution, no box access, no readiness claim.  The audit record names the
mutable owners this scope does *not* isolate (request-row runner storage, the
bypassed row cache, shared per-request KV history, the indexer's call-local
workspace) instead of presenting partial binding as complete isolation.
"""
from contextlib import contextmanager

from e3_types import E3ContractError
from lane_hooks import _ATTR_MISSING, _aliases, LaneRuntime
from lane_chunk import LaneChunk, ChunkAdmission
from model_hooks import local_text_model, ple_embedding

UID_BUFFER_NAME = 'flash_s5_prefill_uid'
TOPK_SCRATCH_NAME = 'prefill_qsa_topk'
UID_ATTR = 'QSA_REQ_UID'
TOPK_ATTR = 'topk_indices_buffer'
VIEW_ALIGNMENT = 16
MAX_MARLIN_LOCK_BYTES = 1024 * 1024
MAX_MARLIN_LOCK_OWNERS = 4096

# Pinned mutable owners that are read/written by the lane path.  ``Topk`` is
# bound privately; the others are named in the audit with their real basis.
_HISTORY_ATTRS = ('kv_cache', '_qsa_host_kv', '_qsa_gpu_slots')
_INDEXER_HISTORY_ATTRS = ('raw_key_cache', 'compressed_key_cache')


class LanePrivateError(E3ContractError):
    pass


def _int(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise LanePrivateError(f'{name} must be an int >= {minimum}')
    return value


def _tensor_of(buffer, name, dtype):
    tensor = getattr(buffer, 'tensor', None)
    if tensor is None:
        raise LanePrivateError(name + ' buffer carries no tensor')
    if getattr(tensor, 'dtype', None) != dtype:
        raise LanePrivateError(name + ' buffer has the wrong dtype')
    if _numel(tensor) < 1:
        raise LanePrivateError(name + ' buffer is empty')
    if getattr(tensor, 'device', None) is None:
        raise LanePrivateError(name + ' buffer has no device')
    _pointer(tensor, name)
    return tensor


def _numel(tensor):
    numel = getattr(tensor, 'numel', None)
    if not callable(numel):
        raise LanePrivateError('tensor has no numel()')
    value = numel()
    if type(value) is not int or value < 0:
        raise LanePrivateError('tensor reports an invalid element count')
    return value


def _element_size(tensor):
    size = getattr(tensor, 'element_size', None)
    if not callable(size):
        raise LanePrivateError('tensor has no element_size()')
    value = size()
    if type(value) is not int or value < 1:
        raise LanePrivateError('tensor reports an invalid element size')
    return value


def _pointer(tensor, name):
    pointer = getattr(tensor, 'data_ptr', None)
    if not callable(pointer):
        raise LanePrivateError(name + ' tensor has no address')
    value = pointer()
    if type(value) is not int or value <= 0:
        raise LanePrivateError(name + ' tensor has an invalid address')
    return value


def _check_alignment(tensor, name):
    pointer = _pointer(tensor, name)
    if pointer % VIEW_ALIGNMENT:
        raise LanePrivateError(
            f'{name} address is not {VIEW_ALIGNMENT}-byte aligned')
    return pointer


def _int32_view(scratch, torch, rows, width, name):
    """Exact-capacity int32 view of an owned uint8 scratch allocation."""
    _int('rows', rows, 1)
    _int('width', width, 1)
    if getattr(scratch, 'dtype', None) != torch.uint8:
        raise LanePrivateError(name + ' is not a raw uint8 allocation')
    element = _element_size(scratch)
    if element != 1:
        raise LanePrivateError(name + ' is not a byte-addressed allocation')
    nbytes = _numel(scratch) * element
    needed = rows * width * 4
    if nbytes < needed:
        raise LanePrivateError(
            name + f' holds {nbytes} bytes; {needed} are required for '
                   f'{rows} x {width} int32 rows')
    if nbytes % 4:
        raise LanePrivateError(name + ' is not divisible into int32 elements')
    _check_alignment(scratch, name)
    if callable(getattr(scratch, 'is_contiguous', None)) and not scratch.is_contiguous():
        raise LanePrivateError(name + ' is not contiguous')
    view = scratch.view(torch.int32)
    view = view[:needed // 4]
    view = view.view(rows, width)
    if tuple(getattr(view, 'shape', ())) != (rows, width):
        raise LanePrivateError(name + ' did not produce the declared view')
    if getattr(view, 'dtype', None) != torch.int32:
        raise LanePrivateError(name + ' view is not int32')
    if not _aliases(view, scratch):
        raise LanePrivateError(name + ' view does not share the owned storage')
    return view


class TopkOwner:
    """One local QSA attention module and the buffer this scope replaces."""

    __slots__ = ('layer_idx', 'layer_name', 'module', 'original', 'width', 'max_tokens')

    def __init__(self, layer_idx, layer_name, module, original, width, max_tokens):
        self.layer_idx = layer_idx
        self.layer_name = layer_name
        self.module = module
        self.original = original
        self.width = width
        self.max_tokens = max_tokens


def _local_layers(model):
    layers = getattr(model, 'layers', None)
    if layers is None or not callable(getattr(layers, '__getitem__', None)):
        raise LanePrivateError('model does not enumerate local layers')
    try:
        length = len(layers)
    except TypeError as exc:
        raise LanePrivateError('model layer container has no length') from exc
    start = _int('model.start_layer', getattr(model, 'start_layer', None))
    end = _int('model.end_layer', getattr(model, 'end_layer', None))
    if end > length:
        raise LanePrivateError('model layer container is shorter than end_layer')
    return tuple(range(start, end))


def _qsa_owner(model, layer_idx, torch):
    layer = model.layers[layer_idx]
    attn = getattr(layer, 'self_attn', None)
    if attn is None:
        if getattr(layer, 'layer_type', None) == 'full_attention':
            raise LanePrivateError('local QSA layer has no attention owner')
        return None
    original = getattr(attn, TOPK_ATTR, _ATTR_MISSING)
    if original is _ATTR_MISSING:
        if getattr(layer, 'layer_type', None) == 'full_attention':
            raise LanePrivateError('local QSA layer has no top-k storage to isolate')
        return None
    if not callable(getattr(attn, '_flash_e3_attention_pre', None)):
        raise LanePrivateError(
            f'layer {layer_idx} carries a pinned QSA top-k buffer but no lane '
            'attention method; the stock path would write decode storage')
    if not getattr(attn, '_qsa_kv_offload', False):
        raise LanePrivateError(
            f'layer {layer_idx} is not an offloaded QSA layer; the lane cannot '
            'bind its top-k buffer through the bounded operation')
    if getattr(attn, '_qsa_host_kv', None) is None:
        raise LanePrivateError(
            f'layer {layer_idx} has no pinned host KV pool')
    layer_name = getattr(attn, 'layer_name', None)
    if not isinstance(layer_name, str) or not layer_name:
        raise LanePrivateError(
            f'layer {layer_idx} has no QSA layer identity')
    indexer = getattr(attn, 'indexer', None)
    width = getattr(indexer, 'packed_output_width', None)
    if type(width) is not int or width < 2:
        raise LanePrivateError(
            f'layer {layer_idx} indexer exposes no packed output width')
    if getattr(original, 'dtype', None) != torch.int32:
        raise LanePrivateError(
            f'layer {layer_idx} top-k buffer is not int32')
    shape = tuple(getattr(original, 'shape', ()))
    if len(shape) != 2 or shape[0] < 1 or shape[1] != width:
        raise LanePrivateError(
            f'layer {layer_idx} top-k shape {shape} disagrees with width {width}')
    _check_alignment(original, f'layer {layer_idx} top-k')
    return TopkOwner(layer_idx, layer_name, attn, original, width, shape[0])


def admitted_chunk(forward_context, *, runner=None):
    """The admitted ``LaneChunk`` behind a lane forward-context callable.

    ``LaneDispatcher`` hands ``chunk.forward_scope`` (a bound context-manager
    method) to ``WorkerNative.lane_forward``.  The chunk, not a generic
    callable, is the authority for the request index and token count.
    """
    chunk = getattr(forward_context, '__self__', None)
    if not isinstance(chunk, LaneChunk):
        raise LanePrivateError(
            'native lane forward needs the admitted chunk as its forward-context callable')
    admission = getattr(chunk, 'admission', None)
    req_index = getattr(chunk, 'req_index', None)
    if admission is None or type(req_index) is not int or req_index < 0:
        raise LanePrivateError('forward context is not an admitted LaneChunk')
    if runner is not None and getattr(chunk, 'runner', None) is not runner:
        raise LanePrivateError('forward context belongs to another runner')
    num_tokens = getattr(admission, 'num_tokens', None)
    if type(num_tokens) is not int or num_tokens < 1:
        raise LanePrivateError('admitted chunk has no positive token count')
    return chunk


class LanePrivateBindings:
    """Per-worker private storage installed for each native lane segment."""

    def __init__(self, runner, runtime, *, chunk_tokens, torch, owners, uid, topk,
                 width, decode_uid, workspace_lanes, ple_scratch, audit):
        self.runner = runner
        self.runtime = runtime
        self.chunk_tokens = chunk_tokens
        self.torch = torch
        self.owners = tuple(owners)
        self.uid = uid
        self.topk = topk
        self.width = width
        self.decode_uid = decode_uid
        self.workspace_lanes = workspace_lanes
        self.ple_scratch = ple_scratch
        self.namespace = runtime.uid_namespace
        self.workspace_lane = runtime.prefill_workspace
        self.audit_record = audit
        self._by_layer = {owner.layer_idx: owner for owner in self.owners}
        self._active = False
        self._failed = False
        self._workspace_ready = False
        self._workspace_buffer = None
        self._workspace_bytes = 0
        self.marlin_locks = []
        self._lock_initialized = set()
        self._ledger = None

    # -- offer provider ----------------------------------------------------
    def attach_ledger(self, ledger):
        """Attach the source-pinned native offer provider (or ``None``).

        ``native_prefill_offers.NativePrefillLedger`` is duck-typed: it owns
        the exact ``private_leaf`` declarations for the initializations this
        scope performs. The hookup is deliberately tiny so the observer
        integration can install it once per worker; without a provider the
        operations run exactly as before.
        """
        if ledger is not None and not callable(getattr(ledger, 'private_leaf', None)):
            raise LanePrivateError(
                'native offer provider exposes no exact private_leaf entry')
        self._ledger = ledger
        return self

    @contextmanager
    def _private_leaf(self, suboperation):
        """Observe one exact initialization leaf; never blocks the original.

        A missing provider, a refused offer or an observation failure latches
        the ledger incomplete (when it has an invalidator) and yields ``None``;
        the caller's body still executes exactly once. The leaf is never
        treated as completion.
        """
        ledger = self._ledger
        if ledger is None:
            yield None
            return
        invalidate = getattr(ledger, '_invalidate', None)

        def invalid(reason):
            # Observer failures cannot skip or duplicate a model operation.
            # The real ledger's invalidator is bounded and non-raising; keep
            # the same body-once contract if a supplied provider itself fails.
            if callable(invalidate):
                try:
                    invalidate(reason)
                except Exception:
                    pass

        opener = getattr(ledger, 'private_leaf', None)
        if not callable(opener):
            invalid('private_leaf_provider_missing')
            yield None
            return
        try:
            scope = opener(suboperation)
            enter, leave = scope.__enter__, scope.__exit__
            meta = enter()
        except Exception:
            invalid('private_leaf_refused:' + str(suboperation))
            yield None
            return
        try:
            yield meta
        except BaseException as exc:
            try:
                leave(type(exc), exc, exc.__traceback__)
            except Exception:
                invalid('private_leaf_close:' + str(suboperation))
            # A provider may observe an operation error, never suppress it.
            raise
        else:
            try:
                leave(None, None, None)
            except Exception:
                invalid('private_leaf_close:' + str(suboperation))

    def _initialize_lock_once(self, private):
        """Zero one retained private lock array on first use, in this scope.

        Extracted so the one-time initialization is auditable and carries its
        exact leaf offer. Order, stream/context, sharing, the one-time flag and
        failure ownership (a raising zero leaves the flag unset) are unchanged.
        """
        with self._private_leaf('lock_init'):
            private.zero_()
        self._lock_initialized.add(id(private))

    # -- discovery ---------------------------------------------------------
    @classmethod
    def discover(cls, runner, runtime, *, chunk_tokens, torch_module, ple_scratch=None):
        if not isinstance(runtime, LaneRuntime):
            raise LanePrivateError('lane-private bindings need the real LaneRuntime')
        _int('chunk_tokens', chunk_tokens, 1)
        if runtime.uid_namespace is None:
            raise LanePrivateError('worker runtime has no QSA UID namespace')
        if runtime.prefill_uid_buffer is None:
            raise LanePrivateError('worker runtime has no private UID buffer')
        if runtime.prefill_context is None:
            raise LanePrivateError('worker runtime has no prefill forward context')
        if runtime.prefill_workspace is None:
            raise LanePrivateError('worker runtime has no prefill workspace lane')

        owned = _owned_buffers(runtime)
        uid_buffer = owned.get(UID_BUFFER_NAME)
        scratch_buffer = owned.get(TOPK_SCRATCH_NAME)
        if uid_buffer is None:
            raise LanePrivateError('worker runtime does not own ' + UID_BUFFER_NAME)
        if scratch_buffer is None:
            raise LanePrivateError('worker runtime does not own ' + TOPK_SCRATCH_NAME)
        uid = _tensor_of(uid_buffer, UID_BUFFER_NAME, torch_module.int32)
        if uid is not runtime.prefill_uid_buffer:
            raise LanePrivateError(
                'runtime private UID buffer is not the owned allocation')
        scratch = _tensor_of(scratch_buffer, TOPK_SCRATCH_NAME, torch_module.uint8)
        if _aliases(uid, scratch):
            raise LanePrivateError('private UID buffer aliases the top-k scratch')

        decode_uid = getattr(runtime.uid_namespace, UID_ATTR, _ATTR_MISSING)
        if decode_uid is _ATTR_MISSING:
            raise LanePrivateError(
                'QSA UID namespace carries no decode buffer to restore')
        if getattr(decode_uid, 'dtype', None) != torch_module.int32:
            raise LanePrivateError('decode QSA UID buffer is not int32')
        if _numel(decode_uid) < 1:
            raise LanePrivateError('decode QSA UID buffer is empty')
        if _aliases(decode_uid, uid):
            raise LanePrivateError('private UID buffer aliases decode storage')
        device = getattr(runner, 'device', None)
        if (not str(device).startswith('cuda:')
                or any(getattr(tensor, 'device', None) != device
                       for tensor in (uid, scratch, decode_uid))):
            raise LanePrivateError('private buffers and decode UID must share the runner CUDA device')
        if len(tuple(getattr(uid, 'shape', ()))) != 1:
            raise LanePrivateError('private batch UID buffer must be one-dimensional')

        lanes = _workspace_lanes(runtime)
        if lanes < 2:
            raise LanePrivateError('prefill workspace lane is not the extra lane')
        if runtime.prefill_workspace != lanes - 1:
            raise LanePrivateError(
                'prefill workspace lane is not the last provisioned lane; '
                'stock lanes and the prefill lane disagree')

        root_model = getattr(runner, 'model', None)
        if root_model is None:
            raise LanePrivateError('runner has no model to bind')
        try:
            model = local_text_model(root_model)
        except E3ContractError as exc:
            raise LanePrivateError('cannot resolve the local text backbone to bind') from exc
        owners = []
        names = set()
        widths = set()
        for layer_idx in _local_layers(model):
            owner = _qsa_owner(model, layer_idx, torch_module)
            if owner is None:
                continue
            if owner.layer_name in names:
                raise LanePrivateError('QSA layer identity is duplicated')
            names.add(owner.layer_name)
            widths.add(owner.width)
            owners.append(owner)
        if len(widths) > 1:
            raise LanePrivateError(
                'local QSA layers disagree on the packed selection width')
        width = next(iter(widths)) if widths else 1
        topk = _int32_view(scratch, torch_module, chunk_tokens, width, TOPK_SCRATCH_NAME)
        if _aliases(topk, decode_uid):
            raise LanePrivateError('private top-k view aliases the decode UID buffer')
        for owner in owners:
            if _aliases(uid, owner.original):
                raise LanePrivateError('private UID buffer aliases decode top-k storage')
            if _aliases(topk, owner.original):
                raise LanePrivateError(
                    'private top-k view aliases decode top-k storage for ' + owner.layer_name)
            if getattr(scratch, 'device', None) != getattr(owner.original, 'device', None):
                raise LanePrivateError(
                    'top-k scratch and the QSA layer live on different devices')

        ple = _ple_audit(model, ple_scratch, chunk_tokens)
        audit = _audit_record(owners, decode_uid, uid, chunk_tokens, width,
                              runtime, lanes, ple, scratch)
        bindings = cls(runner, runtime, chunk_tokens=chunk_tokens, torch=torch_module,
                   owners=owners, uid=uid, topk=topk, width=width,
                   decode_uid=decode_uid, workspace_lanes=lanes,
                   ple_scratch=ple_scratch, audit=audit)
        bindings.marlin_locks = _marlin_locks(model, torch_module, device)
        return bindings

    # -- per-chunk ---------------------------------------------------------
    def chunk_scope(self, chunk):
        return LanePrivateChunk(self, chunk)

    def audit(self):
        return dict(self.audit_record, failed=self._failed,
                    workspace_prepared=self._workspace_ready,
                    workspace_bytes=self._workspace_bytes,
                    marlin_lock_owners=len(self.marlin_locks),
                    marlin_lock_bytes=sum(_numel(t) * _element_size(t)
                        for t in {id(row[4]): row[4] for row in self.marlin_locks}.values()))

    def prepare_workspace(self):
        """Fund the reserved lane after stock warmup, before capture's lock.

        The pinned profile_run exercises max_num_tokens; this native chunk is
        no larger, or, above 1024 rows (dp5), funded in proportion. Mirror the actual stock workspace high-water allocation into
        the already-provisioned extra lane. Never resize or replace stock graph
        storage, unlock a manager, or allow lazy live allocation after capture.
        """
        module = self.runtime.dispatcher.workspace.module
        manager = module.current_workspace_manager()
        if self._workspace_ready:
            self._require_workspace()
            return self._workspace_bytes
        maximum = getattr(self.runner, 'max_num_tokens', None)
        # decode-phase dp5: above 1024 rows the native chunk may exceed stock's token bound, which stays 1024; its lane is funded in proportion below.
        if type(maximum) is not int or maximum < 1 or (maximum < self.chunk_tokens and self.chunk_tokens <= 1024):
            raise LanePrivateError('stock profile token bound does not cover the native chunk')
        if (getattr(manager, '_num_ubatches', None) != 1
                or getattr(manager, '_num_lanes', None) != self.workspace_lanes
                or getattr(manager, '_locked', None) is not False):
            raise LanePrivateError('prefill workspace must be funded before the single-ubatch manager locks')
        slots = getattr(manager, '_current_workspaces', None)
        if type(slots) is not list or len(slots) != self.workspace_lanes:
            raise LanePrivateError('workspace allocation slots are not the pinned layout')
        stock = slots[0]
        if stock is None:
            raise LanePrivateError('stock workspace warmup did not provide a capacity bound')
        needed = _numel(stock) * _element_size(stock)
        if needed < 1:
            raise LanePrivateError('stock workspace warmup has no allocated capacity')
        if self.chunk_tokens > maximum:
            needed = -(-needed * self.chunk_tokens // maximum)
        capturing = getattr(self.torch.cuda, 'is_current_stream_capturing', None)
        if not callable(capturing) or capturing():
            raise LanePrivateError('prefill workspace funding requires a non-capturing startup stream')
        allocate = getattr(manager, '_ensure_workspace_size', None)
        if not callable(allocate):
            raise LanePrivateError('workspace manager has no pinned allocation entry')
        with module.use_workspace_lane(self.workspace_lane):
            private = allocate(needed)
        if (slots[0] is not stock or private is not slots[self.workspace_lane]
                or _aliases(private, stock)
                or _numel(private) * _element_size(private) < needed
                or getattr(private, 'device', None) != getattr(stock, 'device', None)):
            raise LanePrivateError('prefill workspace funding changed or aliases stock storage')
        self._workspace_manager = manager
        self._workspace_buffer = private
        self._workspace_bytes = _numel(private) * _element_size(private)
        self._workspace_ready = True
        return self._workspace_bytes

    def _require_workspace(self):
        manager = self.runtime.dispatcher.workspace.module.current_workspace_manager()
        if (not self._workspace_ready or manager is not self._workspace_manager
                or manager._current_workspaces[self.workspace_lane] is not self._workspace_buffer):
            raise LanePrivateError('prefill workspace has no retained funded allocation')

    def _enter(self):
        if self._failed:
            raise LanePrivateError('private binding restoration failed; ownership quarantined')
        self._require_workspace()
        if self._active:
            raise LanePrivateError('another lane-private scope is active')
        self._active = True

    def _exit(self):
        self._active = False

    def _owner_for(self, layer_idx):
        if layer_idx is None:
            return None
        return self._by_layer.get(layer_idx)


class LanePrivateChunk:
    """The bindings for one admitted chunk; one scope per native segment."""

    def __init__(self, bindings, chunk):
        if not isinstance(bindings, LanePrivateBindings):
            raise LanePrivateError('chunk scope needs the worker bindings')
        if not isinstance(chunk, LaneChunk) or getattr(chunk, 'runner', None) is not bindings.runner:
            raise LanePrivateError('chunk belongs to another runner')
        req_index = getattr(chunk, 'req_index', None)
        if type(req_index) is not int or req_index < 0:
            raise LanePrivateError('chunk has no admitted request index')
        if req_index >= _numel(bindings.uid):
            raise LanePrivateError('admitted request index exceeds the private UID buffer')
        admission = getattr(chunk, 'admission', None)
        if type(admission) is not ChunkAdmission:
            raise LanePrivateError('private scope needs the canonical chunk admission')
        num_tokens = getattr(admission, 'num_tokens', None)
        if type(num_tokens) is not int or num_tokens < 1:
            raise LanePrivateError('chunk has no positive token count')
        if num_tokens > bindings.chunk_tokens:
            raise LanePrivateError(
                f'chunk of {num_tokens} tokens exceeds the {bindings.chunk_tokens}-token '
                'top-k scratch funding')
        self.bindings = bindings
        self.chunk = chunk
        self.req_index = req_index
        self.num_tokens = num_tokens
        self.admission = admission
        self.handle = bindings.runtime.prefill_handle
        self._populated = False

    def _require_current(self):
        runtime = self.bindings.runtime
        if (self.chunk.admission is not self.admission
                or runtime.prefill_txn != self.admission.transaction
                or self.handle is None or runtime.prefill_handle is not self.handle
                or runtime.state.prefill.handle is not self.handle
                or runtime.state.prefill.state not in ('ready', 'in_use')):
            raise LanePrivateError('private scope names a stale or closed admission')
        runtime.state.check_transaction(self.admission.transaction)
        states = getattr(self.bindings.runner, 'req_states', None)
        rows = getattr(states, 'req_id_to_index', None)
        if rows is None or rows.get(self.admission.request_uid) != self.req_index:
            raise LanePrivateError('admitted request no longer owns its runner row')

    # -- scope -------------------------------------------------------------
    @contextmanager
    def scope(self, layer_idx=None):
        bindings = self.bindings
        bindings._enter()
        try:
            self._require_current()
            with bindings.runtime.dispatcher.enter(
                    None, bindings.runtime.prefill_context,
                    bindings.runtime.prefill_workspace):
                saved = []
                try:
                    current_uid = getattr(bindings.namespace, UID_ATTR, _ATTR_MISSING)
                    if current_uid is not bindings.decode_uid:
                        raise LanePrivateError(
                            'QSA UID namespace changed since discovery; refusing')
                    self._populate_uid_once()
                    bindings.namespace.QSA_REQ_UID = bindings.uid
                    saved.append((bindings.namespace, UID_ATTR, bindings.decode_uid))
                    owner = bindings._owner_for(layer_idx)
                    if owner is not None:
                        if getattr(owner.module.indexer, 'skip_topk', False):
                            raise LanePrivateError('native prefill cannot reuse decoder top-k selection')
                        current = getattr(owner.module, TOPK_ATTR, _ATTR_MISSING)
                        if current is not owner.original:
                            raise LanePrivateError(
                                'QSA top-k owner changed since discovery; refusing')
                        if _aliases(current, bindings.topk):
                            raise LanePrivateError(
                                'private top-k view aliases decode storage')
                        setattr(owner.module, TOPK_ATTR, bindings.topk)
                        saved.append((owner.module, TOPK_ATTR, owner.original))
                    # Kernels retain tensor pointers, not Python attributes. Keep
                    # private lock allocations for the worker lifetime, swap only
                    # the host owner during this segment, and initialize on this
                    # same prefill stream before any auxiliary consumer launches.
                    for layer, path, module, original, private in bindings.marlin_locks:
                        if layer != layer_idx:
                            continue
                        if getattr(module, 'workspace', None) is not original:
                            raise LanePrivateError('Marlin lock owner changed since discovery')
                        if id(private) not in bindings._lock_initialized:
                            bindings._initialize_lock_once(private)
                        setattr(module, 'workspace', private)
                        saved.append((module, 'workspace', original))
                    yield self
                finally:
                    restore_error = None
                    for owner, name, previous in reversed(saved):
                        try:
                            if previous is _ATTR_MISSING:
                                if hasattr(owner, name):
                                    delattr(owner, name)
                            else:
                                setattr(owner, name, previous)
                        except BaseException as exc:
                            restore_error = restore_error or exc
                    if restore_error is not None:
                        bindings._failed = True
                        bindings.runtime.dispatcher.failed = True
                        bindings.runtime.state.prefill.quarantine('private owner restoration failed')
                        raise LanePrivateError('private owner restoration failed; storage retained') from None
        finally:
            bindings._exit()

    # -- request UID -------------------------------------------------------
    def _populate_uid_once(self):
        if self._populated:
            return
        bindings = self.bindings
        runner = bindings.runner
        slots = getattr(runner, '_qsa_uid_slots', None)
        if slots is None:
            raise LanePrivateError('worker runner carries no QSA UID slot table')
        if getattr(slots, 'dtype', None) != bindings.torch.int32:
            raise LanePrivateError('QSA UID slot table is not int32')
        if getattr(slots, 'device', None) != getattr(bindings.uid, 'device', None):
            raise LanePrivateError('QSA UID slot table lives on another device')
        if len(tuple(getattr(slots, 'shape', ()))) != 1:
            raise LanePrivateError('canonical QSA UID slots must be one-dimensional')
        if _aliases(slots, bindings.uid) or _aliases(slots, bindings.topk):
            raise LanePrivateError('canonical QSA UID slots alias private lane scratch')
        if self.req_index >= _numel(slots):
            raise LanePrivateError(
                'admitted request index exceeds the QSA UID slot table')
        issued = getattr(runner, '_qsa_uid_next', 0)
        if type(issued) is not int or issued < 1:
            raise LanePrivateError('admitted request has no issued QSA UID')
        # One device-to-device copy on the lane stream; no host read, no sync.
        # The fill and the seed copy are the exact uid_init leaf; a missing
        # provider or refused offer latches incomplete but still runs both
        # operations once, in this order.
        with bindings._private_leaf('uid_init'):
            bindings.uid.fill_(-2)
            bindings.uid[0].copy_(slots[self.req_index])
        self._populated = True


# --- discovery helpers ------------------------------------------------------

def _marlin_locks(model, torch, device):
    """Pinned INC -> scheme -> linear_method -> kernel workspace owners.

    Generic WorkspaceManager does not own Marlin's persistent int32 lock
    arrays (mixed_precision/marlin.py:95,200). Also retain layer.workspace
    used by quantized MoE variants. Unknown workspace geometry is refused.
    Call-local MoE locks (marlin_moe.py:97-98) are separate fresh allocations.
    """
    rows, allocations, seen = [], [], set()
    total = 0
    for layer_idx in _local_layers(model):
        layer = model.layers[layer_idx]
        modules = getattr(layer, 'named_modules', None)
        if not callable(modules):
            raise LanePrivateError('local layer cannot enumerate mutable quantization owners')
        for name, module in modules():
            pending = [(name, module, 0)]
            visited = set()
            while pending:
                path, owner, depth = pending.pop()
                if owner is None or id(owner) in visited:
                    continue
                visited.add(id(owner))
                if depth > 5:
                    raise LanePrivateError('quantization workspace owner chain is not pinned')
                for attr in ('quant_method', 'scheme', 'linear_method', 'kernel'):
                    child = getattr(owner, attr, None)
                    if child is not None:
                        pending.append((path + '.' + attr, child, depth + 1))
                original = getattr(owner, 'workspace', None)
                if original is None:
                    continue
                key = (layer_idx, id(owner))
                if key in seen:
                    continue
                seen.add(key)
                if (getattr(original, 'dtype', None) != torch.int32
                        or getattr(original, 'device', None) != device
                        or len(tuple(getattr(original, 'shape', ()))) != 1
                        or not 0 < _numel(original) <= 4096
                        or not callable(getattr(original, 'is_contiguous', None))
                        or not original.is_contiguous()):
                    raise LanePrivateError('persistent quantization workspace is not a bounded int32 lock array')
                private = None
                for old, new in allocations:
                    if _aliases(original, old):
                        if (_pointer(original, 'lock') != _pointer(old, 'lock')
                                or tuple(original.shape) != tuple(old.shape)):
                            raise LanePrivateError('persistent lock arrays partially overlap')
                        private = new
                        break
                if private is None:
                    size = _numel(original) * _element_size(original)
                    total += size
                    if total > MAX_MARLIN_LOCK_BYTES:
                        raise LanePrivateError('persistent Marlin lock funding exceeds the declared cap')
                    private = torch.empty(tuple(original.shape), dtype=original.dtype,
                                          device=device)
                    if (getattr(private, 'device', None) != device
                            or getattr(private, 'dtype', None) != original.dtype
                            or tuple(private.shape) != tuple(original.shape)
                            or any(_aliases(private, old) or _aliases(private, new)
                                   for old, new in allocations)
                            or _aliases(private, original)):
                        raise LanePrivateError('persistent Marlin lock allocation aliases or differs')
                    allocations.append((original, private))
                rows.append((layer_idx, path, owner, original, private))
                if len(rows) > MAX_MARLIN_LOCK_OWNERS:
                    raise LanePrivateError('persistent Marlin lock owner count exceeds the declared cap')
    # The same host owner may occur in several layers; keep each activation.
    return tuple(rows)


def _owned_buffers(runtime):
    owned = getattr(runtime, '_native_owned_buffers', None)
    if owned is None:
        raise LanePrivateError('worker runtime does not retain its owned buffers')
    by_name = {}
    for buffer in owned:
        name = getattr(buffer, 'name', None)
        if not isinstance(name, str) or not name:
            raise LanePrivateError('owned buffer has no name')
        if name in by_name:
            raise LanePrivateError('owned buffer name is duplicated: ' + name)
        by_name[name] = buffer
    return by_name


def _workspace_lanes(runtime):
    adapter = getattr(getattr(runtime, 'dispatcher', None), 'workspace', None)
    module = getattr(adapter, 'module', None)
    lane = getattr(adapter, 'lane', None)
    if module is None or type(lane) is not int:
        raise LanePrivateError('workspace adapter does not expose its lane')
    if lane != runtime.prefill_workspace:
        raise LanePrivateError('workspace adapter and runtime disagree on the prefill lane')
    getter = getattr(module, 'current_workspace_manager', None)
    if not callable(getter):
        raise LanePrivateError('workspace module cannot report its manager')
    try:
        manager = getter()
    except Exception as exc:
        raise LanePrivateError('workspace manager is not initialized') from exc
    lanes = getattr(manager, '_num_lanes', None)
    if type(lanes) is not int or lanes < 1:
        raise LanePrivateError('workspace manager reports no lane count')
    return lanes


def _ple_audit(model, ple_scratch, chunk_tokens=1024):
    found = []
    for layer_idx in _local_layers(model):
        embedding = ple_embedding(model, layer_idx, chunk_tokens, getattr(ple_scratch, 'selectable', False))
        if embedding is not None:
            found.append((layer_idx, embedding))
    if not found:
        return {'present': False}
    if len(found) > 1:
        raise LanePrivateError('more than one PLE layer on this stage is not supported')
    layer_idx, embedding = found[0]
    if ple_scratch is None or getattr(ple_scratch, 'embedding', None) is not embedding:
        raise LanePrivateError(
            'PLE prefetch buffer/stream has no private lane owner')
    return {'present': True, 'layer': layer_idx,
            'binding': 'private_ple_scratch',
            'stream_basis': getattr(ple_scratch, 'stream_basis', 'unknown')}


def _audit_record(owners, decode_uid, uid, chunk_tokens, width, runtime,
                  lanes, ple, scratch):
    layers = []
    for owner in owners:
        module = owner.module
        indexer = getattr(module, 'indexer', None)
        layers.append({
            'layer': owner.layer_idx,
            'name': owner.layer_name,
            'topk': 'private_view',
            'topk_rows': chunk_tokens,
            'topk_width': width,
            'decode_topk_rows': owner.max_tokens,
            'rowcache': ('present_bypassed'
                         if getattr(module, '_qsa_rowcache', None) is not None
                         else 'absent'),
            'history': [name for name in _HISTORY_ATTRS
                        if getattr(module, name, None) is not None],
            'indexer_history': [name for name in _INDEXER_HISTORY_ATTRS
                                if getattr(indexer, name, None) is not None],
            'skip_topk': bool(getattr(indexer, 'skip_topk', False)),
        })
    unbound = [
        'runner.input_buffers/model_state/block_tables request rows are written '
        'at admission outside this scope and are not address-isolated here',
        'QSA row cache is bypassed by the lane driver, not bound',
        'per-request KV history is shared under the existing lifetime proof',
        'the pinned indexer allocates its selection workspace per call; it is not '
        'funded by this inventory',
    ]
    return {
        'isolation': 'partial',
        'topk_scratch_bytes': _numel(scratch) * _element_size(scratch),
        'topk_view': {'shape': [chunk_tokens, width], 'dtype': 'int32',
                      'alignment': VIEW_ALIGNMENT},
        'topk_reuse': ('one QSA layer at a time; the operation proves consumer '
                       'completion before the next layer writes the scratch'),
        'uid_rows': _numel(uid),
        'decode_uid_rows': _numel(decode_uid),
        'workspace_lane': runtime.prefill_workspace,
        'workspace_lanes': lanes,
        'workspace_basis': 'extra_last_lane',
        'ple_prefetch': ple,
        'qsa_layers': layers,
        'unbound': unbound,
        'readiness': 'unclaimed',
    }


__all__ = ['LanePrivateBindings', 'LanePrivateChunk', 'LanePrivateError',
           'TOPK_ATTR', 'TOPK_SCRATCH_NAME', 'UID_ATTR', 'UID_BUFFER_NAME',
           'VIEW_ALIGNMENT', 'admitted_chunk']
