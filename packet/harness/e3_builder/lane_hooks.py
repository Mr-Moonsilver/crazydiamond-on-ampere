"""E3 runner seams: exact pinned-source transforms plus bounded CPU runtime.

The transform targets the approved v2 runner, whose bytes are pinned by
``PINNED_SHA256``. In this repository that exact file is committed as
``patches/rowcache/model_runner_v2.py.patched`` (sha256 verified against
``reference/vllm-src/v1/worker/gpu/model_runner.py``); the reference tree itself
is not needed to generate or test the transforms.

Every replacement has a unique anchor. ``generate`` fails closed on an
unpinned source, an existing E3 marker or a missing/duplicated anchor;
``revert`` recovers the original bytes exactly. E2 composition is checked
against deterministic reviewed transforms, not import/decorator markers. The
transforms are narrow insertions/one rename. Apply E2's ``runner_execution``
first and E3 second: E2 then decorates the renamed stock method and E3's
wrapper routes to it. Applying E3 first is not supported because E2's anchor
would land inside the E3 wrapper. C owns the final ``make_s5_overlap.py``
composition and package paths.

Runtime seams use bounded credits: the supplied backend must itself make
``submit`` and ``poll`` bounded and nonblocking (C's integration responsibility);
``commit`` requires C's completion proof. No torch/vLLM import happens here.
"""
import ast
from contextlib import contextmanager
import hashlib
import os
import threading

from e3_types import (
    E3ContractError,
    E3Pending,
    LaneRole,
    OperationTicket,
    PollResult,
    TransactionId,
)
from lane_state import BatchRow, MetadataSnapshot, WorkerState

PINNED_SHA256 = 'a49588ad1ab173b1b5b1695d28fa64da7bbfc5a13d7d32d7aaaf9781879069a2'

WARMUP_ID_PREFIXES = ('_warmup_', '_v2_mixed_warmup')
_ATTR_MISSING = object()


def warmup_request_ids(output):
    """All request ids a frameless execute call names; never logged."""
    ids = []
    for request in getattr(output, 'scheduled_new_reqs', None) or ():
        ids.append(getattr(request, 'req_id', None))
    cached = getattr(output, 'scheduled_cached_reqs', None)
    ids.extend(getattr(cached, 'req_ids', None) or ())
    ids.extend((getattr(output, 'num_scheduled_tokens', None) or {}).keys())
    ids.extend(getattr(output, 'finished_req_ids', None) or ())
    return tuple(ids)


# Reviewed E2 transforms of the approved v2 runner. Keep full-file pins even
# when generator code runs from a bundle without the repository reference tree.
E2_EXECUTION_SHA256 = 'ff701a20769ed9d13bfe8ae986531d79c24e46cb812ef1e8f029427a093bde93'
E2_STARTUP_SHA256 = '65f1c6da7918159bd360a7fda4de6f8d24b53e27af706adf6ec3b9c127a027c6'


def source_kind(source):
    """Accept only the pinned baseline or the two reviewed E2 compositions."""
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest == PINNED_SHA256:
        return 'pinned'
    if digest in (E2_EXECUTION_SHA256, E2_STARTUP_SHA256):
        return 'e2'
    raise ValueError('unreviewed model_runner source: ' + digest)


def _require_pinned(source):
    if source_kind(source) != 'pinned':
        raise ValueError('source is not the pinned model_runner bytes')
    return source


def _replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('source anchor changed or duplicated: ' + old[:80].replace('\n', '\\n'))
    return text.replace(old, new, 1)


_IMPORT = '''from vllm.forward_context import BatchDescriptor, set_forward_context
'''

_IMPORT_E3 = '''from vllm.forward_context import BatchDescriptor, set_forward_context

import vllm.v1.flash_e3.lane_hooks as _flash_e3_hooks
'''

_WRAPPER = '''    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        return _flash_e3_hooks.execute_with_e3(
            self,
            self._flash_e3_stock_execute_model,
            scheduler_output,
            intermediate_tensors,
            dummy_run,
            skip_attn_for_dummy_run,
            is_profile,
            context_len,
            valid_dummy_state_slots,
        )

    @torch.inference_mode()
    @step_eplb_after()
    def sample_tokens(
'''

# Ordered forward replacements. Each ``new`` is inserted once; revert applies
# the inverse in reverse order and must recover the pinned bytes.
FORWARD = (
    (_IMPORT, _IMPORT_E3),
    ('    def execute_model(\n', '    def _flash_e3_stock_execute_model(\n'),
    ('    @torch.inference_mode()\n    @step_eplb_after()\n    def sample_tokens(\n', _WRAPPER),
    ('''        num_tokens = batch_req_state.num_tokens
        num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)
''', '''        _flash_e3_hooks.before_prepare_inputs(self, scheduler_output, batch_req_state)
        num_tokens = batch_req_state.num_tokens
        num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)
'''),
    ('''        self._qsa_write_req_uids(input_batch)
''', '''        if not _flash_e3_hooks.prepare_attn_uids(self, input_batch):
            self._qsa_write_req_uids(input_batch)
'''),
    ('''        slots = getattr(self, "_qsa_uid_slots", None)
''', '''        if _flash_e3_hooks.new_uid(self, req_index):
            return
        slots = getattr(self, "_qsa_uid_slots", None)
'''),
    ('''    def _qsa_write_req_uids(self, input_batch: InputBatch) -> None:
        buf = getattr(_qsa_fc, "QSA_REQ_UID", None)
''', '''    def _qsa_write_req_uids(self, input_batch: InputBatch) -> None:
        if _flash_e3_hooks.write_uids(self, input_batch):
            return
        buf = getattr(_qsa_fc, "QSA_REQ_UID", None)
'''),
    ('''        copy_kv_cache_blocks_inplace(views, num_blocks, copies)
        self._qsa_srcs.update(c.src_block_id for c in copies)
''', '''        copy_kv_cache_blocks_inplace(views, num_blocks, copies)
        _flash_e3_hooks.host_copies(self, copies, views, num_blocks)
        self._qsa_srcs.update(c.src_block_id for c in copies)
'''),
    ('''        self.model_state.remove_request(req_id)
''', '''        _flash_e3_hooks.check_remove(self, req_id)
        self.model_state.remove_request(req_id)
'''),
    ('''            req_id = new_req_data.req_id
''', '''            req_id = new_req_data.req_id
            _flash_e3_hooks.check_add(self, new_req_data)
'''),
    ('''    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
''', '''    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        _flash_e3_hooks.check_update(self, scheduler_output)
'''),
    ('''        return batch_state, get_uniform_decode_token_count(
''', '''        _flash_e3_hooks.note_batch(self, scheduler_output, batch_state)
        return batch_state, get_uniform_decode_token_count(
'''),
    ('''    ) -> AsyncOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
''', '''    ) -> AsyncOutput | ModelRunnerOutput | None:
        _flash_e3_hooks.check_sample_tokens(self)
        if self.execute_model_state is None:
'''),
    ('''    def postprocess_num_computed_tokens(self, input_batch: InputBatch) -> None:
        # Update the number of computed tokens.
        post_update_num_computed_tokens(
''', '''    def postprocess_num_computed_tokens(self, input_batch: InputBatch) -> None:
        if _flash_e3_hooks.defer_num_computed(self, input_batch):
            return
        # Update the number of computed tokens.
        post_update_num_computed_tokens(
'''),
    ('''            self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)
''', '''            self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)
        _flash_e3_hooks.init_lane(self, kv_caches_dict, is_profiling)
'''),
)


def generate(source):
    """Return the E3-transformed runner source (pinned or E2-composed input)."""
    source_kind(source)
    if '_flash_e3' in source:
        raise ValueError('runner already carries E3 hooks')
    text = source
    for old, new in FORWARD:
        text = _replace(text, old, new)
    if '_install_flash_e2(self, is_profiling)' in source:
        line = '        _flash_e3_hooks.init_lane(self, kv_caches_dict, is_profiling)\n'
        text = _replace(text, line, '')
        text = _replace(text, '        _install_flash_e2(self, is_profiling)\n',
                        '        _install_flash_e2(self, is_profiling)\n' + line)
    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise ValueError('generated runner syntax invalid') from exc
    return text


def revert(source, expected=None):
    """Reverse the E3-only transform and prove the recovered bytes.

    With no ``expected`` argument the recovery must be the pinned source. For
    an E3-on-E2 composition pass the E2-composed input as ``expected``; C's
    generator owns the full-file pin for that path.
    """
    if '_flash_e3' not in source:
        raise ValueError('source does not carry E3 hooks')
    text = source
    if '_install_flash_e2(self, is_profiling)' in text:
        line = '        _flash_e3_hooks.init_lane(self, kv_caches_dict, is_profiling)\n'
        text = _replace(text, line, '')
        anchor = '            self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)\n'
        text = _replace(text, anchor, anchor + line)
    for old, new in reversed(FORWARD):
        text = _replace(text, new, old)
    if expected is None:
        _require_pinned(text)
    elif text != expected:
        raise ValueError('revert did not recover the supplied source')
    return text


class _NullBackend:
    """A bounded backend that never completes anything."""

    def submit(self, ticket):
        return None

    def poll(self, transaction):
        return ()


class HostDispatcher:
    """Single owner of process-global forward context and workspace state.

    The caller must not have two Python forwards active. Entry installs the lane
    context/workspace; exit restores the previous values even on exception.
    """

    def __init__(self, forward_context, workspace):
        if forward_context is None or workspace is None:
            raise E3ContractError('dispatcher needs forward-context and workspace adapters')
        self.forward_context = forward_context
        self.workspace = workspace
        self.active = False
        self.owner_thread = threading.get_ident()

    def assert_owner(self):
        if threading.get_ident() != self.owner_thread:
            raise E3ContractError('lane dispatcher called from a foreign host thread')

    @contextmanager
    def enter(self, lane, context, workspace):
        self.assert_owner()
        if getattr(self, 'failed', False):
            raise E3ContractError('dispatcher context restoration failed; quarantined')
        if self.active:
            raise E3Pending('another Python forward context is installed')
        previous_context = self.forward_context.get()
        self.active = True
        entered = False
        try:
            self.forward_context.set(context)
            # Workspace adapters must support rollback after a partial enter.
            entered = True
            self.workspace.enter(workspace)
            yield
        finally:
            try:
                if entered:
                    self.workspace.exit(workspace)
            except BaseException:
                self.failed = True
                raise
            finally:
                try:
                    self.forward_context.set(previous_context)
                except BaseException:
                    self.failed = True
                    raise
                finally:
                    self.active = False


class AttributeBinding:
    """Owned private module/runner storage installed only by the host dispatcher.

    C's startup builder supplies the actual object and private allocation. A
    restores decode's attribute and retains any lazy replacement for the next
    prefill submission. This is not a clone or an allocation/funding claim.
    """
    def __init__(self, owner, name, value):
        if owner is None or not isinstance(name, str) or not name or name.startswith('__'):
            raise E3ContractError('invalid mutable attribute target')
        self.owner, self.name, self.value = owner, name, value


def _aliases(a, b):
    if a is b:
        return True
    # Native tensor storage is inspected without a CUDA read or torch import.
    # Distinct views of one allocation are still shared storage.
    if hasattr(a, 'untyped_storage') and hasattr(b, 'untyped_storage'):
        sa, sb = a.untyped_storage(), b.untyped_storage()
        if getattr(a, 'device', None) == getattr(b, 'device', None):
            lo, hi = sa.data_ptr(), sa.data_ptr() + sa.nbytes()
            blo, bhi = sb.data_ptr(), sb.data_ptr() + sb.nbytes()
            return max(lo, blo) < min(hi, bhi)
    return False


class LaneRuntime:
    """Bounded A/C interface for one worker's decode+prefill lane state."""

    def __init__(self, state, *, forward_context, workspace, submit_backend=None,
                 prefill_forward_context=None, prefill_workspace=None, uid_namespace=None, prefill_uid_buffer=None):
        if type(state) is not WorkerState:
            raise E3ContractError('runtime needs WorkerState')
        self.state = state
        self.dispatcher = HostDispatcher(forward_context, workspace)
        self.backend = submit_backend if submit_backend is not None else _NullBackend()
        self.prefill_context = prefill_forward_context
        self.prefill_workspace = prefill_workspace
        self.uid_namespace = uid_namespace
        self.prefill_uid_buffer = prefill_uid_buffer
        self.prefill_handle = None
        self.prefill_txn = None
        self.prefill_execution = None
        self.private_attrs = {}
        self.object_bindings = ()
        self.real_frames = 0
        self.warmup_rpcs = 0
        self._preparing_prefill = False
        self._prefill_running = False
        self._pending_num_computed = None
        self._host_copies = []
        self._committed = None

    # --- admission / bounded preparation ---------------------------------
    def open_prefill(self, transaction, metadata, *, bindings=(), private_attrs=None,
                     context_id=None, object_bindings=()):
        self.dispatcher.assert_owner()
        if type(metadata) is not MetadataSnapshot:
            raise E3ContractError('prefill admission needs an immutable metadata snapshot')
        if private_attrs is not None and not hasattr(private_attrs, 'items'):
            raise E3ContractError('private_attrs must be a mapping')
        if '_qsa_uid_slots' in (private_attrs or {}):
            raise E3ContractError('request UID slots are canonical; only the batch UID buffer is lane-private')
        handle = self.state.prefill.reserve(transaction, metadata=metadata,
                                            bindings=bindings, context_id=context_id)
        self.state.prefill.ready()
        self.state.prefill.note_batch(metadata)
        self.prefill_handle = handle
        self.prefill_txn = transaction
        self.prefill_execution = None
        self.private_attrs = dict(private_attrs or {})
        self.object_bindings = tuple(object_bindings)
        if any(type(b) is not AttributeBinding for b in self.object_bindings):
            self.state.prefill.quarantine('invalid module binding')
            raise E3ContractError('module storage requires AttributeBinding')
        return handle

    def prepare_prefill(self, runner, scheduler_output, batch_req_state, batch_desc,
                        metadata=None):
        """Prepare one private prefill batch; no forward and no publication."""
        lane = self._prefill_lane()
        if not lane.outstanding:
            raise E3ContractError('prefill preparation requires a registered completion ticket')
        if metadata is not None:
            lane.note_batch(metadata)
        if self.prefill_context is None or self.prefill_workspace is None:
            raise E3ContractError('prefill context/workspace are not bound')
        with self.dispatcher.enter(self.prefill_handle, self.prefill_context,
                                   self.prefill_workspace):
            with self._swap_prefill(runner):
                self._preparing_prefill = True
                try:
                    return runner.prepare_inputs(scheduler_output, batch_req_state, batch_desc)
                finally:
                    self._preparing_prefill = False

    # --- bounded submit / completion query -------------------------------
    def submit(self, role, transaction, op_seq, dependency_role, backend=None):
        if backend is not None and backend is not self.backend:
            raise E3ContractError('submission and polling must use the same owned backend')
        lane = self._lane_for(transaction, role)
        ticket = OperationTicket(transaction, self.state.worker_rank, lane.lane_id,
                                 op_seq, dependency_role)
        lane.submit(ticket)
        try:
            (backend or self.backend).submit(ticket)
        except BaseException:
            lane.quarantine('submit raised')
            raise
        return ticket

    def poll(self, transaction):
        """Nonblocking completion query. Never synchronizes or waits."""
        lane = self._query_lane(transaction)
        if lane.failed:
            return PollResult((), len(lane.outstanding), True)
        try:
            ready = (self.backend.poll(transaction) or ())
        except BaseException:
            lane.quarantine('poll raised')
            raise
        if not isinstance(ready, (tuple, list)) or len(ready) > self.state.max_outstanding:
            lane.quarantine('backend poll returned an unbounded batch')
            raise E3ContractError('backend poll must return a bounded completion batch')
        completed = []
        try:
            for proof in ready:
                lane.complete(proof)
                completed.append(proof)
        except BaseException:
            lane.quarantine('invalid completion batch')
            raise
        return PollResult(tuple(completed), len(lane.outstanding),
                          lane.failed)

    # --- commit / abort / release (C owns the policy) --------------------
    def propose_state(self, transaction, value):
        self._prefill_lane(transaction).persistent.propose(transaction, value)

    def commit(self, transaction, proof):
        record = self._prefill_lane(transaction).persistent.commit(transaction, proof)
        self._committed = record
        self.prefill_execution = None
        return record

    def abort(self, transaction):
        self._prefill_lane(transaction).abort('transaction aborted')
        # Retain outputs and private objects until every GPU/PP consumer drains.

    def release(self, transaction, proofs=()):
        released = self._prefill_lane(transaction).release(proofs)
        self.prefill_handle = None
        self.prefill_txn = None
        self.prefill_execution = None
        self.private_attrs = {}
        self.object_bindings = ()
        self._pending_num_computed = None
        return released

    def fail_completion(self, transaction, ticket, reason='failed completion'):
        lane = self._lane_for(transaction, None)
        lane.fail(ticket, reason)
        return lane._quarantine

    # --- runner call routing ---------------------------------------------
    def execute(self, runner, stock, scheduler_output, transaction, *args):
        self.dispatcher.assert_owner()
        if self.dispatcher.active:
            raise E3Pending('another Python forward has installed its context')
        self.state.check_transaction(transaction)
        if transaction == self.prefill_txn:
            return self._execute_prefill(runner, stock, scheduler_output, *args)
        self._lane_for(transaction, LaneRole.DECODE)
        self.real_frames += 1
        return stock(scheduler_output, *args)

    def execute_native_decode(self, runner, stock, scheduler_output, *args):
        """Ordinary decode under the persistent owner: no request transaction.

        B's graph hooks make the typed retained-storage route decision and
        replay the actual split bank; this method only preserves A's single
        forward/dispatcher ownership.
        """
        self.dispatcher.assert_owner()
        if self.dispatcher.active:
            raise E3Pending('another Python forward has installed its context')
        self.real_frames += 1
        return stock(scheduler_output, *args)

    def _execute_prefill(self, runner, stock, scheduler_output, *args):
        lane = self._prefill_lane()
        if not lane.outstanding:
            raise E3ContractError('prefill execution requires a registered completion ticket')
        if self.prefill_execution is not None:
            raise E3Pending('prefill execution state is still owned')
        if self.prefill_context is None or self.prefill_workspace is None:
            raise E3ContractError('prefill context/workspace are not bound')
        self.real_frames += 1
        with self.dispatcher.enter(self.prefill_handle, self.prefill_context,
                                   self.prefill_workspace):
            with self._swap_prefill(runner):
                self._prefill_running = True
                saved_state = getattr(runner, 'execute_model_state', _ATTR_MISSING)
                runner.execute_model_state = None
                try:
                    output = stock(scheduler_output, *args)
                    pending = runner.execute_model_state
                    if pending is None:
                        raise E3ContractError('prefill forward produced no execution state')
                    # Lane-local slot: never decode's pending execute_model_state.
                    self.prefill_execution = pending
                except BaseException:
                    self.prefill_execution = getattr(runner, 'execute_model_state', None)
                    lane.quarantine('prefill execution failed; storage retained')
                    raise
                finally:
                    self._prefill_running = False
                    if saved_state is _ATTR_MISSING:
                        del runner.execute_model_state
                    else:
                        runner.execute_model_state = saved_state
                return output

    @contextmanager
    def _swap_prefill(self, runner):
        saved = []
        if self.uid_namespace is None or self.prefill_uid_buffer is None:
            raise E3ContractError('prefill needs the actual QSA UID namespace and a private batch buffer')
        previous_uid = getattr(self.uid_namespace, 'QSA_REQ_UID', _ATTR_MISSING)
        if previous_uid is _ATTR_MISSING or _aliases(previous_uid, self.prefill_uid_buffer):
            raise E3ContractError('prefill UID buffer must be distinct from the decode buffer')
        bindings = [AttributeBinding(runner, name, value) for name, value in self.private_attrs.items()]
        bindings += list(self.object_bindings)
        keys = [(id(b.owner), b.name) for b in bindings]
        if len(set(keys)) != len(keys):
            raise E3ContractError('duplicate mutable attribute binding')
        self.uid_namespace.QSA_REQ_UID = self.prefill_uid_buffer
        try:
            for binding in bindings:
                previous = getattr(binding.owner, binding.name, _ATTR_MISSING)
                if previous is not _ATTR_MISSING and _aliases(previous, binding.value):
                    raise E3ContractError('prefill attribute aliases decode storage')
                saved.append((binding, previous))
                setattr(binding.owner, binding.name, binding.value)
            yield
        finally:
            try:
                for binding, previous in reversed(saved):
                    # Keep lazily replaced scratch alive after restoring decode.
                    binding.value = getattr(binding.owner, binding.name, binding.value)
                    if binding.owner is runner and binding.name in self.private_attrs:
                        self.private_attrs[binding.name] = binding.value
                    if previous is _ATTR_MISSING:
                        if hasattr(binding.owner, binding.name):
                            delattr(binding.owner, binding.name)
                    else:
                        setattr(binding.owner, binding.name, previous)
            finally:
                self.uid_namespace.QSA_REQ_UID = previous_uid

    # --- source-seam helpers ---------------------------------------------
    def is_vllm_warmup(self, output):
        if self.real_frames:
            return False
        if getattr(output, 'flash_e2_frame', None) is not None:
            return False
        if getattr(output, 'flash_e3_transaction', None) is not None:
            return False
        ids = warmup_request_ids(output)
        return bool(ids) and all(
            isinstance(request_id, str) and request_id.startswith(WARMUP_ID_PREFIXES)
            for request_id in ids)

    @contextmanager
    def warmup(self, output):
        if not self.is_vllm_warmup(output):
            raise E3ContractError('warmup pass-through needs vLLM warmup request ids')
        self.warmup_rpcs += 1
        try:
            yield
        except BaseException:
            self.dispatcher.failed = True
            raise

    def before_prepare_inputs(self, runner, scheduler_output, batch_req_state):
        if self.prefill_txn is None:
            return
        if self._prefill_batch():
            self._validate_batch(batch_req_state)
        elif getattr(runner, 'execute_model_state', None) is not None:
            raise E3Pending('decode is awaiting sample_tokens')

    def note_batch(self, runner, scheduler_output, batch_state):
        if self.prefill_txn is None:
            return None
        if getattr(scheduler_output, 'flash_e3_transaction', None) != self.prefill_txn:
            return None
        self._validate_batch(batch_state)
        return self.state.prefill._latest_batch

    def _validate_batch(self, batch):
        snapshot = self.state.prefill.metadata
        if snapshot is None:
            raise E3ContractError('missing immutable admission metadata')
        if tuple(batch.req_ids) != tuple(row.request_uid for row in snapshot.rows):
            raise E3ContractError('stale batch row')
        if snapshot.slot_mappings and tuple(int(i) for i in batch.idx_mapping_np) != snapshot.slot_mappings:
            raise E3ContractError('stale batch row')
        computed = getattr(batch, 'num_computed_prefill_tokens_np', None)
        if snapshot.num_computed and (computed is None or tuple(int(i) for i in computed) != snapshot.num_computed):
            raise E3ContractError('computed frontier differs from admission')

    def _prefill_batch(self):
        return self._preparing_prefill or self._prefill_running

    def prepare_attn_uids(self, runner, input_batch):
        # Stock writer gathers canonical per-request UIDs into the lane-local
        # batch buffer installed in the actual QSA forward-context namespace.
        return False

    def new_uid(self, runner, req_index):
        # Never restart UID allocation per lane/chunk: row-cache ownership is
        # shared and requires the runner's canonical monotonic request identity.
        return False

    def write_uids(self, runner, input_batch):
        return False

    def host_copies(self, runner, copies, views, num_blocks):
        if self.prefill_txn is None:
            return None
        record = (len(copies), len(views), int(num_blocks))
        self._host_copies.append(record)
        return record

    def check_remove(self, runner, req_id):
        if self.prefill_txn is None:
            return
        if req_id == self.prefill_txn.request_uid:
            raise E3Pending('retire the active prefill only through C completion/commit')

    def check_add(self, runner, new_req_data):
        if self.prefill_txn is None:
            return
        req_id = getattr(new_req_data, 'req_id', None)
        generation = getattr(new_req_data, 'request_generation', None)
        if generation is not None and req_id in self.state._generations:
            if generation != self.state.current_generation(req_id):
                raise E3ContractError('request generation is stale')

    def check_update(self, runner, scheduler_output):
        if self.prefill_txn is None or not self.state.prefill.outstanding:
            return
        # This guard runs BEFORE any CPU table write, zeroing or CoW. C can
        # validate a disjoint mutation against real scheduler source leases;
        # absent that proof refuse mutations conservatively, never after copying.
        cached = getattr(scheduler_output, 'scheduled_cached_reqs', None)
        mutates = bool(getattr(scheduler_output, 'block_table_updates', None)
                       or getattr(scheduler_output, 'new_block_ids_to_zero', None)
                       or getattr(scheduler_output, 'kv_cache_block_copies', None)
                       or any(x is not None for x in (getattr(cached, 'new_block_ids', ()) or ())))
        if mutates:
            guard = getattr(self, 'mutation_guard', None)
            if guard is None or guard(scheduler_output, self.prefill_txn) is not True:
                raise E3Pending('worker mutation needs scheduler lease proof before update')

    def check_sample_tokens(self, runner):
        self.dispatcher.assert_owner()
        current = getattr(runner, 'execute_model_state', None)
        if self._prefill_running or (self.prefill_execution is not None and current is self.prefill_execution):
            raise E3ContractError('ordinary sample_tokens cannot consume a prefill execution')

    def defer_num_computed(self, runner, input_batch):
        if self.prefill_txn is None or not self._prefill_batch():
            return False
        indices = getattr(input_batch, 'idx_mapping_np', ())
        offsets = getattr(input_batch, 'query_start_loc_np', ())
        self._pending_num_computed = (tuple(int(i) for i in indices),
                                      tuple(int(i) for i in offsets))
        return True

    def _lane_for(self, transaction, role):
        self.dispatcher.assert_owner()
        if role is LaneRole.PREFILL:
            return self._prefill_lane(transaction)
        if role is LaneRole.DECODE:
            lane = self.state.decode
            lane._require_active(transaction)
            return lane
        if role is None:
            if transaction == self.prefill_txn:
                return self._prefill_lane(transaction)
            lane = self.state.decode
            lane._require_active(transaction)
            return lane
        raise E3ContractError('unknown lane role')

    def _query_lane(self, transaction):
        """Resolve a leased lane for polling, including a quarantined one."""
        self.dispatcher.assert_owner()
        lane = self.state.prefill if transaction == self.prefill_txn else self.state.decode
        if lane.handle is None or lane.transaction != transaction:
            raise E3ContractError('transaction does not own an active lane')
        return lane

    def _prefill_lane(self, transaction=None):
        self.dispatcher.assert_owner()
        lane = self.state.prefill
        lane._require_active(transaction)
        if self.prefill_handle is None or lane.handle != self.prefill_handle:
            raise E3ContractError('prefill handle is stale')
        return lane


def _batch_row(request_uid, generation, index, transaction):
    return BatchRow(request_uid, generation, index,
                    transaction.chunk_start, transaction.chunk_end, 0)


def _runtime(runner):
    runtime = getattr(runner, '_flash_e3_lane', None)
    if runtime is not None and not isinstance(runtime, LaneRuntime):
        raise E3ContractError('runner E3 lane is not a LaneRuntime')
    return runtime


def native_decode_ready(runner):
    """True only when native startup installed B's retained-storage decode path.

    The decode-only diagnostic sets a DecodeProbeRouter (with no
    ``native_decode_route``) and is deliberately excluded.
    """
    if not getattr(runner, '_flash_s5_native_decode', False):
        return False
    router = getattr(runner, '_flash_e3_graph_router', None)
    return router is not None and callable(getattr(router, 'native_decode_route', None))


def execute_with_e3(runner, stock, scheduler_output, intermediate_tensors=None,
                    dummy_run=False, skip_attn_for_dummy_run=False, is_profile=False,
                    context_len=0, valid_dummy_state_slots=False):
    """Generated ``execute_model`` wrapper. Off/dummy/profile is byte-identical."""
    runtime = _runtime(runner)
    forwarded = (intermediate_tensors, dummy_run, skip_attn_for_dummy_run,
                 is_profile, context_len, valid_dummy_state_slots)
    if runtime is not None and getattr(runtime.dispatcher, 'failed', False):
        raise E3ContractError('E3 dispatcher is quarantined')
    if runtime is None or dummy_run or is_profile:
        return stock(scheduler_output, *forwarded)
    if skip_attn_for_dummy_run or context_len or valid_dummy_state_slots:
        raise E3ContractError('profiling/capture flags on a real E3 RPC')
    transaction = getattr(scheduler_output, 'flash_e3_transaction', None)
    if transaction is None:
        if runtime.is_vllm_warmup(scheduler_output):
            with runtime.warmup(scheduler_output):
                return stock(scheduler_output, *forwarded)
        if native_decode_ready(runner):
            # Persistent decode storage: no request transaction or per-step
            # ticket.  B decides stock vs the actual split bank and refuses an
            # ambiguous route rather than falling back.
            return runtime.execute_native_decode(runner, stock, scheduler_output, *forwarded)
        raise E3ContractError('real frameless E3 work is not permitted')
    if type(transaction) is not TransactionId:
        raise E3ContractError('scheduler output carries an invalid E3 transaction')
    return runtime.execute(runner, stock, scheduler_output, transaction, *forwarded)


# Narrow seams called by generated source. Each is inert unless C installed a
# runtime, so the default-off path retains stock behaviour.
def before_prepare_inputs(runner, scheduler_output, batch_req_state):
    runtime = _runtime(runner)
    if runtime is not None:
        runtime.before_prepare_inputs(runner, scheduler_output, batch_req_state)


def prepare_attn_uids(runner, input_batch):
    runtime = _runtime(runner)
    return runtime.prepare_attn_uids(runner, input_batch) if runtime is not None else False


def new_uid(runner, req_index):
    runtime = _runtime(runner)
    return runtime.new_uid(runner, req_index) if runtime is not None else False


def write_uids(runner, input_batch):
    runtime = _runtime(runner)
    return runtime.write_uids(runner, input_batch) if runtime is not None else False


def host_copies(runner, copies, views, num_blocks):
    runtime = _runtime(runner)
    if runtime is not None:
        return runtime.host_copies(runner, copies, views, num_blocks)
    return None


def check_remove(runner, req_id):
    runtime = _runtime(runner)
    if runtime is not None:
        runtime.check_remove(runner, req_id)


def check_add(runner, new_req_data):
    runtime = _runtime(runner)
    if runtime is not None:
        runtime.check_add(runner, new_req_data)


def check_update(runner, scheduler_output):
    runtime = _runtime(runner)
    if runtime is not None:
        runtime.check_update(runner, scheduler_output)


def note_batch(runner, scheduler_output, batch_state):
    runtime = _runtime(runner)
    if runtime is not None:
        return runtime.note_batch(runner, scheduler_output, batch_state)
    return None


def check_sample_tokens(runner):
    runtime = _runtime(runner)
    if runtime is not None:
        runtime.check_sample_tokens(runner)


def defer_num_computed(runner, input_batch):
    runtime = _runtime(runner)
    return runtime.defer_num_computed(runner, input_batch) if runtime is not None else False


def init_lane(runner, kv_caches, is_profiling=False):
    """Startup seam. Off is inert; on requires C's builder; never a stub."""
    if is_profiling or os.environ.get('FLASH_S5_OVERLAP', '0') != '1':
        return None
    existing = getattr(runner, '_flash_e3_lane', None)
    if existing is not None:
        raise E3ContractError('E3 lane already installed')
    builder = getattr(runner, '_flash_e3_build_state', None)
    if builder is None:
        raise E3ContractError('FLASH_S5_OVERLAP=1 without C startup builder')
    runtime = builder(runner, kv_caches)
    if not isinstance(runtime, LaneRuntime):
        raise E3ContractError('C startup builder did not return LaneRuntime')
    if isinstance(runtime.backend, _NullBackend):
        raise E3ContractError('enabled runtime requires C completion backend, not the null fixture')
    if runtime.uid_namespace is None or runtime.prefill_uid_buffer is None:
        raise E3ContractError('enabled runtime requires native QSA UID buffer bindings')
    runner._flash_e3_lane = runtime
    return runtime


__all__ = [
    'AttributeBinding', 'FORWARD', 'HostDispatcher', 'LaneRuntime', 'PINNED_SHA256', 'WARMUP_ID_PREFIXES',
    'before_prepare_inputs', 'check_add', 'check_remove', 'check_sample_tokens',
    'check_update', 'defer_num_computed', 'execute_with_e3', 'generate', 'host_copies',
    'init_lane', 'native_decode_ready', 'new_uid', 'note_batch', 'prepare_attn_uids',
    'revert', 'warmup_request_ids', 'write_uids',
]
