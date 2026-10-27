"""CPU-only tests for the E3 lane-private mutable bindings (package I).

The fakes below are address-bearing tensors: every view keeps the storage it
was cut from, so alias, capacity and restoration checks exercise the same
predicates as the native path.  ``LaneForward`` and ``QsaLayerDriver`` are the
real modules; only the model/attention/operation objects are fixtures.
"""
import contextlib
import ast
import inspect
import hashlib
import os
from contextvars import ContextVar
from itertools import accumulate
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
sys.path.insert(0, str(ROOT / 'harness'))

import lane_private  # noqa: E402
from e3_types import E3Pending  # noqa: E402
from lane_hooks import LaneRuntime  # noqa: E402
from lane_private import LanePrivateBindings, LanePrivateError, admitted_chunk  # noqa: E402
from lane_state import WorkerState
from lane_chunk import LaneChunk, ChunkAdmission
from test_e3_lane_state import metadata as lane_metadata
from dataclasses import replace  # noqa: E402
from model_hooks import LaneForward  # noqa: E402
from native_resources import _ForwardContextAdapter, _WorkspaceLaneAdapter  # noqa: E402
from qsa_driver import QsaLayerDriver  # noqa: E402

_ELEMENT = {'uint8': 1, 'int32': 4, 'bf16': 2, 'float32': 4}
_NEXT_ADDRESS = [0x200000]


def _numel(shape):
    total = 1
    for size in shape:
        total *= size
    return total


class Storage:
    """One allocation: a base address, byte length and byte contents."""

    def __init__(self, nbytes, base=None):
        if base is None:
            _NEXT_ADDRESS[0] += 0x1000
            base = _NEXT_ADDRESS[0]
        self._base = base
        self._nbytes = nbytes
        self.buf = bytearray(nbytes)

    def data_ptr(self):
        return self._base

    def nbytes(self):
        return self._nbytes


class Tensor:
    """A view over :class:`Storage` with int32 byte access for assertions."""

    def __init__(self, shape, dtype, *, storage=None, offset=0, device='cuda:0'):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.offset = offset
        self._storage = storage or Storage(_numel(self.shape) * _ELEMENT[dtype])

    # -- tensor protocol ---------------------------------------------------
    def numel(self):
        return _numel(self.shape)

    def element_size(self):
        return _ELEMENT[self.dtype]

    def data_ptr(self):
        return self._storage.data_ptr() + self.offset

    def untyped_storage(self):
        return self._storage

    def is_contiguous(self):
        return True

    def view(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            if self.numel() % 4:
                raise ValueError('size is not divisible by four')
            return Tensor((self.numel() // 4,), args[0], storage=self._storage,
                          offset=self.offset, device=self.device)
        shape = args[0] if len(args) == 1 and isinstance(args[0], tuple) else tuple(args)
        if _numel(shape) != self.numel():
            raise ValueError('shape view changes the element count')
        return Tensor(shape, self.dtype, storage=self._storage, offset=self.offset,
                      device=self.device)

    def __getitem__(self, item):
        if isinstance(item, int):
            index = item if item >= 0 else self.shape[0] + item
            if not 0 <= index < self.shape[0]:
                raise IndexError(index)
            stride = self.element_size()
            for size in self.shape[1:]:
                stride *= size
            return Tensor((), self.dtype, storage=self._storage,
                          offset=self.offset + index * stride, device=self.device)
        if isinstance(item, slice):
            start, stop, _ = item.indices(self.shape[0])
            stride = self.element_size()
            for size in self.shape[1:]:
                stride *= size
            shape = (stop - start,) + self.shape[1:]
            return Tensor(shape, self.dtype, storage=self._storage,
                          offset=self.offset + start * stride, device=self.device)
        raise TypeError('unsupported index')

    def fill_(self, value):
        if self.dtype != 'int32':
            raise TypeError('only int32 fill is modelled')
        start = self.offset
        end = start + self.numel() * 4
        self._storage.buf[start:end] = int(value).to_bytes(4, 'little', signed=True) * self.numel()
        return self

    def zero_(self):
        size = self.numel() * self.element_size()
        self._storage.buf[self.offset:self.offset + size] = b'\x00' * size
        return self

    def copy_(self, other, non_blocking=False):
        size = min(self.numel(), other.numel()) * 4
        if self.dtype != 'int32' or other.dtype != 'int32':
            raise TypeError('only int32 copies are modelled')
        start = other.offset
        self._storage.buf[self.offset:self.offset + size] = \
            other._storage.buf[start:start + size]
        return self

    # -- test helpers ------------------------------------------------------
    def read(self, index=0):
        start = self.offset + index * 4
        return int.from_bytes(self._storage.buf[start:start + 4], 'little', signed=True)

    def write(self, index, value):
        start = self.offset + index * 4
        self._storage.buf[start:start + 4] = int(value).to_bytes(4, 'little', signed=True)

    def row_width(self):
        return self.shape[1] if len(self.shape) > 1 else 1

    def write_row(self, row, value):
        width = self.row_width()
        for column in range(width):
            self.write(row * width + column, value)

    def read_row(self, row):
        width = self.row_width()
        return [self.read(row * width + column) for column in range(width)]

    def fill_rows(self, base):
        for row in range(self.shape[0]):
            self.write_row(row, base + row)


class FakeStream:
    def __init__(self, token):
        self.cuda_stream = token


class FakeEvent:
    def __init__(self, enable_timing=False):
        self.stream = None

    def record(self, stream=None):
        self.stream = stream


class FakeCuda:
    def __init__(self):
        self._stack = [FakeStream(1)]
        self.Event = FakeEvent

    def is_current_stream_capturing(self):
        return False

    def current_stream(self, device=None):
        return self._stack[-1]

    @contextlib.contextmanager
    def stream(self, stream):
        self._stack.append(stream)
        try:
            yield stream
        finally:
            self._stack.pop()


def fake_torch():
    torch = types.SimpleNamespace()
    torch.uint8 = 'uint8'
    torch.int32 = 'int32'
    torch.cuda = FakeCuda()
    torch.empty = lambda shape, dtype, device: Tensor(shape, dtype, device=device)
    torch.accelerator = types.SimpleNamespace(empty_cache=lambda: None)
    return torch


class FakeWorkspaceModule:
    """Actual pinned WorkspaceManager; only CUDA allocation is replaced."""
    def __init__(self, lanes, torch):
        self.active = None
        self.entered = []
        self.lane_var = ContextVar('test_workspace_lane', default=0)
        source = (ROOT / 'reference/vllm-src/v1/worker/workspace.py').read_text()
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'WorkspaceManager')
        namespace = dict(torch=torch, _workspace_lane=self.lane_var,
                         dbo_current_ubatch_id=lambda: 0, inspect=inspect, os=os,
                         envs=types.SimpleNamespace(VLLM_DEBUG_WORKSPACE=False),
                         logger=types.SimpleNamespace(info=lambda *a: None),
                         _MB=1024*1024, accumulate=accumulate,
                         round_up=lambda n, d: ((n+d-1)//d)*d,
                         _compute_bytes=lambda shape, dtype: _numel(shape)*_ELEMENT[dtype])
        text = 'from __future__ import annotations\n' + ast.get_source_segment(source, cls)
        exec(compile(text, 'pinned-workspace-manager', 'exec'), namespace)
        self.manager = namespace['WorkspaceManager']('cuda:0', 1, lanes)

    @property
    def lanes(self):
        return self.manager._num_lanes

    @lanes.setter
    def lanes(self, value):
        self.manager._num_lanes = value

    def current_workspace_manager(self):
        return self.manager

    @contextlib.contextmanager
    def use_workspace_lane(self, lane):
        previous = self.active
        token = self.lane_var.set(lane)
        self.active = lane
        self.entered.append(('enter', lane))
        try:
            yield
        finally:
            self.entered.append(('exit', lane))
            self.active = previous
            self.lane_var.reset(token)


class FakeForwardContextModule:
    def __init__(self, decode_uid):
        self._forward_context = None
        self.QSA_REQ_UID = decode_uid


class FakeIndexer:
    def __init__(self, width):
        self.packed_output_width = width
        self.raw_key_cache = types.SimpleNamespace(prefix='side.raw')
        self.compressed_key_cache = types.SimpleNamespace(prefix='side.compressed')
        self.skip_topk = False
        self.calls = []
        self.fail = False

    def __call__(self, projected_qk, positions, out):
        if self.fail:
            raise RuntimeError('indexer refused the private buffer')
        if out.shape[1] != self.packed_output_width:
            raise AssertionError('private buffer width changed')
        self.calls.append(out)
        out.fill_rows(501)
        return out


class FakeImpl:
    def __init__(self):
        self.updates = 0

    def do_kv_cache_update(self, attn, key, value, kv_cache, slot_mapping):
        self.updates += 1


class FakeQsaAttention:
    _qsa_kv_offload = True
    _max_decode_query_len = 1

    def __init__(self, width, rows=16):
        self.layer_name = 'model.layers.0.attn'
        self.topk_indices_buffer = Tensor((rows, width), 'int32')
        self.indexer = FakeIndexer(width)
        self.impl = FakeImpl()
        self.kv_cache = Tensor((8,), 'uint8')
        self._qsa_host_kv = object()
        self._qsa_rowcache = None
        self._qsa_gpu_slots = object()

    def _flash_e3_attention_pre(self, positions, hidden):
        num_tokens = hidden.shape[0]
        return {'query': Tensor((num_tokens, 4), 'bf16'),
                'key': Tensor((num_tokens, 4), 'bf16'),
                'value': Tensor((num_tokens, 4), 'bf16'),
                'gate': None,
                'attn_output': Tensor((num_tokens, 4), 'bf16'),
                'projected_qk': Tensor((num_tokens, 4), 'bf16'),
                'num_tokens': num_tokens}

    def _flash_e3_attention_post(self, state):
        return Tensor((state['num_tokens'], 4), 'bf16')


class FakeLayer:
    def __init__(self, attn, tokens):
        self.layer_type = 'full_attention'
        self.self_attn = attn
        self.ple = None
        self.tokens = tokens

    def named_modules(self):
        return (('', self), ('self_attn', self.self_attn))

    def _flash_e3_layer_pre(self, hidden, block_output, injection, *, input_ids=None,
                            query_start_loc=None, ngram_context=None):
        return ('pre-hidden', Tensor((self.tokens, 4), 'bf16'), 'injection')

    def _flash_e3_layer_post(self, hidden, attn_out, injection):
        return ('next-hidden', 'block', 'injection')


class FakeModel:
    def __init__(self, layer):
        self.layers = [layer]
        self.start_layer = 0
        self.end_layer = 1

    def _flash_e3_forward_prefix(self, input_ids, intermediate_tensors, inputs_embeds):
        return 'prefix-hidden'

    def _flash_e3_forward_layer(self, layer_idx, *args):
        raise AssertionError('QSA layer must run through the driver')

    def _flash_e3_forward_suffix(self, *args):
        return {'hidden_states': 'done'}

    def _flash_e3_is_last_stage(self):
        return False


class FakeChunk(LaneChunk):
    """The admitted-chunk surface LaneDispatcher hands to lane_forward."""

    def __init__(self, runner, req_index, num_tokens, metadata):
        self.runner = runner
        self.req_index = req_index
        tx = replace(runner._test_transaction,
                     chunk_end=runner._test_transaction.chunk_start + num_tokens)
        self.admission = ChunkAdmission(tx, tx.chunk_end + 1, (1,) * num_tokens,
                                        (), ((1, 2),), state_epoch=1)
        self.forward_context = types.SimpleNamespace(attn_metadata=metadata)

    @contextlib.contextmanager
    def forward_scope(self):
        module = self.runner._fc_module
        previous = module._forward_context
        module._forward_context = self.forward_context
        try:
            yield self.forward_context
        finally:
            module._forward_context = previous

    def lane_inputs(self):
        return dict(positions='positions', input_ids='input_ids',
                    query_start_loc='query_start_loc', ngram_context='ngram')


class FakeOperation:
    """Real driver protocol with one pending bucket then completion."""

    def __init__(self, *, pending=1, probe=None):
        self.pending = pending
        self.probe = probe
        self.stage = 0
        self.generator = None
        self.ended = []
        self.observations = []

    def begin_layer(self, layer_name, host_kv):
        self._observe('begin', layer_name)

    def producer_submitted(self, layer_name, key, value, kv_cache):
        self._observe('producer')

    def attach_continuation(self, generator):
        self.generator = generator

    def advance(self):
        self._observe('advance')
        if self.stage < self.pending:
            self.stage += 1
            raise E3Pending('bucket pending')
        raise StopIteration(self.stage)

    def end_layer(self, layer_name):
        self.ended.append(layer_name)
        self._observe('end')

    def cancel(self, reason='cancelled'):
        return reason

    def _observe(self, phase, layer_name=None):
        if self.probe is not None:
            self.observations.append(self.probe(phase, self, layer_name))


def staged_operation(operation, attn, query, kv_cache, logical_indices, metadata,
                     num_tokens, use_prefill_config, out):
    """The driver's captured selection view survives the pause, not the attribute."""
    return types.SimpleNamespace(logical_indices=logical_indices, out=out,
                                 num_tokens=num_tokens)


def metadata_for(attn, num_tokens):
    main = types.SimpleNamespace(num_actual_tokens=num_tokens,
                                 max_query_len=num_tokens,
                                 slot_mapping='slots')
    side = types.SimpleNamespace(num_actual_tokens=num_tokens)
    return {attn.layer_name: main, 'side.raw': side}


class PrivateBindingsTests(unittest.TestCase):
    chunk_tokens = 8
    width = 4
    req_index = 2

    def setUp(self):
        self.torch = fake_torch()
        self.attn = FakeQsaAttention(self.width)
        self.original = self.attn.topk_indices_buffer
        self.original.fill_rows(900)
        self.model = FakeModel(FakeLayer(self.attn, self.chunk_tokens))
        self.decode_uid = Tensor((4,), 'int32')
        self.decode_uid.fill_(-2)
        self.private_uid = Tensor((4,), 'int32')
        self.private_uid.fill_(-2)
        self.scratch = Tensor((self.chunk_tokens * self.width * 4,), 'uint8')
        self.slots = Tensor((4,), 'int32')
        self.slots.fill_(-2)
        self.slots.write(self.req_index, 7)
        self.fc = FakeForwardContextModule(self.decode_uid)
        self.ws = FakeWorkspaceModule(lanes=2, torch=self.torch)
        self.runtime = LaneRuntime(
            WorkerState(0, engine_epoch=3),
            forward_context=_ForwardContextAdapter(self.fc),
            workspace=_WorkspaceLaneAdapter(self.ws, 1),
            prefill_forward_context=object(), prefill_workspace=1,
            uid_namespace=self.fc, prefill_uid_buffer=self.private_uid)
        self.runtime._native_owned_buffers = (
            types.SimpleNamespace(name=lane_private.UID_BUFFER_NAME, tensor=self.private_uid),
            types.SimpleNamespace(name=lane_private.TOPK_SCRATCH_NAME, tensor=self.scratch))
        self.runner = types.SimpleNamespace(
            model=self.model, device='cuda:0', _fc_module=self.fc,
            max_num_tokens=self.chunk_tokens,
            _qsa_uid_slots=self.slots, _qsa_uid_next=7)
        tx, snapshot = lane_metadata(self.runtime.state, end=self.chunk_tokens)
        self.runtime.open_prefill(tx, snapshot)
        self.runner._test_transaction = tx
        self.runner.req_states = types.SimpleNamespace(req_id_to_index={tx.request_uid: self.req_index})
        self.releases = []
        runtime = self.runtime
        self.runtime.release = lambda transaction, proofs=(): (
            self.releases.append(transaction), None)[1]
        self.ws.manager._ensure_workspace_size(128)
        self.bindings = self.discover()
        self.bindings.prepare_workspace()
        self.ws.manager.lock()
        self.workspace_start = len(self.ws.entered)

    def discover(self, **overrides):
        kwargs = dict(chunk_tokens=self.chunk_tokens, torch_module=self.torch)
        kwargs.update(overrides)
        return LanePrivateBindings.discover(self.runner, self.runtime, **kwargs)

    def make_chunk(self, num_tokens=None):
        num_tokens = self.chunk_tokens if num_tokens is None else num_tokens
        return FakeChunk(self.runner, self.req_index, num_tokens,
                         metadata_for(self.attn, num_tokens))

    def make_lane(self, chunk, *, bindings=None, pending=1, probe=None):
        operation = FakeOperation(pending=pending, probe=probe)
        driver = QsaLayerDriver(operation, staged_operation=staged_operation,
                                forward_context=lambda: chunk.forward_context,
                                torch_module=self.torch)
        scope = None
        if bindings is not None:
            scope = bindings.chunk_scope(chunk).scope
        lane = LaneForward(self.model, chunk.lane_inputs(), stream=FakeStream(700),
                           forward_context=chunk.forward_scope,
                           attention_driver=driver, torch_module=self.torch,
                           clock=lambda: 0, private_scope=scope)
        return lane, operation

    def probe_factory(self, bindings):
        def probe(phase, operation, layer_name=None):
            return {
                'phase': phase,
                'stage': operation.stage,
                'uid_is_private': self.fc.QSA_REQ_UID is self.private_uid,
                'uid_value': self.private_uid.read(0),
                'workspace': self.ws.active,
                'topk_is_private': self.attn.topk_indices_buffer is bindings.topk,
                'topk_row': bindings.topk.read_row(0)[0],
            }
        return probe

    # -- the interleaved prefill/decode case ------------------------------
    def test_interleaved_paused_prefill_never_writes_decode_storage(self):
        chunk = self.make_chunk()
        probe = self.probe_factory(self.bindings)
        lane, operation = self.make_lane(chunk, bindings=self.bindings, probe=probe)
        lane.begin()
        self.assertEqual(self.original.read_row(0)[0], 900)

        # Segment 1: the attention pre-phase selects into the private view.
        record = lane.layer(0)
        self.assertEqual(record.kind, 'layer-paused')
        self.assertIs(self.attn.topk_indices_buffer, self.original)
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)
        self.assertIsNone(self.ws.active)
        self.assertIsNone(self.fc._forward_context)
        self.assertEqual(self.bindings.topk.read_row(0)[0], 501)
        self.assertEqual(self.original.read_row(0)[0], 900)

        # Decode owns its captured address between segments and may write it.
        self.original.write_row(0, 901)

        # Segment 2: the paused generator still holds the private selection.
        record = lane.layer(0)
        self.assertEqual(record.kind, 'layer-paused')
        observed = operation.observations
        advance = [item for item in observed if item['phase'] == 'advance'][0]
        self.assertTrue(advance['uid_is_private'])
        self.assertEqual(advance['uid_value'], 7)
        self.assertEqual(advance['workspace'], 1)
        self.assertTrue(advance['topk_is_private'])
        self.assertEqual(advance['topk_row'], 501)
        self.assertIsNotNone(operation.generator)
        self.assertEqual(operation.generator.logical_indices.read_row(0)[0], 501)
        self.assertIs(self.attn.topk_indices_buffer, self.original)
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)
        self.assertIsNone(self.ws.active)
        self.assertEqual(self.original.read_row(0)[0], 901)

        # Segment 3: completion restores the exact decode owners again.
        record = lane.layer(0)
        self.assertEqual(record.kind, 'layer')
        self.assertEqual(operation.ended, [self.attn.layer_name])
        self.assertIs(self.attn.topk_indices_buffer, self.original)
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)
        self.assertEqual(self.original.read_row(0)[0], 901)
        lane.finish()

        self.assertEqual(self.releases, [])
        self.assertIs(self.bindings.topk._storage, self.scratch._storage)
        self.assertEqual([event for event, _ in self.ws.entered[self.workspace_start:]].count('enter'), 5)
        self.assertEqual([event for event, _ in self.ws.entered[self.workspace_start:]].count('exit'), 5)

    def test_missing_private_scope_writes_the_captured_decode_buffer(self):
        # Counterexample for the delivered gap: no private_scope means the
        # driver writes the tensor decode's graph captured.
        chunk = self.make_chunk()
        lane, operation = self.make_lane(chunk, bindings=None)
        lane.begin()
        self.assertEqual(self.original.read_row(0)[0], 900)
        lane.layer(0)
        self.assertEqual(self.original.read_row(0)[0], 501)
        self.assertEqual(self.attn.topk_indices_buffer.read_row(0)[0], 501)
        self.assertIsNone(self.ws.active)
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)

    # -- restoration and refusals -----------------------------------------
    def test_canonical_uid_slots_must_not_alias_private_batch_uid(self):
        self.runner._qsa_uid_slots = self.private_uid
        with self.assertRaises(LanePrivateError):
            with self.bindings.chunk_scope(self.make_chunk()).scope(0):
                pass

    def test_private_uid_and_topk_must_share_the_runner_device(self):
        self.private_uid.device = 'cuda:1'
        with self.assertRaises(LanePrivateError):
            self.discover()

    def test_chunk_scope_refuses_another_active_admission(self):
        scope = self.bindings.chunk_scope(self.make_chunk())
        self.runtime.prefill_txn = object()
        with self.assertRaises(LanePrivateError):
            with scope.scope(0):
                pass

    def test_private_scope_rechecks_request_row_and_retained_storage(self):
        scope = self.bindings.chunk_scope(self.make_chunk())
        self.runner.req_states.req_id_to_index[self.runner._test_transaction.request_uid] += 1
        with self.assertRaisesRegex(LanePrivateError, 'runner row'):
            with scope.scope(0):
                pass
        self.runner.req_states.req_id_to_index[self.runner._test_transaction.request_uid] = self.req_index
        self.runtime.prefill_handle = None
        with self.assertRaisesRegex(LanePrivateError, 'stale or closed'):
            with scope.scope(0):
                pass

    def test_restoration_failure_restores_other_owners_and_quarantines(self):
        scope = self.bindings.chunk_scope(self.make_chunk())
        setter = FakeQsaAttention.__setattr__
        def refuse_restore(owner, name, value):
            if owner is self.attn and name == 'topk_indices_buffer' and value is self.original:
                raise RuntimeError('controlled restoration fault')
            return setter(owner, name, value)
        with patch.object(FakeQsaAttention, '__setattr__', refuse_restore):
            with self.assertRaisesRegex(LanePrivateError, 'storage retained'):
                with scope.scope(0):
                    self.assertIs(self.attn.topk_indices_buffer, self.bindings.topk)
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)
        self.assertIsNone(self.ws.active)
        self.assertFalse(self.bindings._active)
        self.assertTrue(self.runtime.state.prefill.failed)
        self.assertTrue(self.bindings.audit()['failed'])
        with self.assertRaisesRegex(LanePrivateError, 'quarantined'):
            with scope.scope(0):
                pass

    def test_missing_qsa_owner_and_decoder_selection_reuse_refuse(self):
        del self.attn.topk_indices_buffer
        with self.assertRaisesRegex(LanePrivateError, 'no top-k storage'):
            self.discover()
        self.attn.topk_indices_buffer = self.original
        self.attn.indexer.skip_topk = True
        with self.assertRaisesRegex(LanePrivateError, 'reuse decoder'):
            with self.bindings.chunk_scope(self.make_chunk()).scope(0):
                pass
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)

    def test_lookalike_bound_method_is_not_an_admitted_chunk(self):
        class Lookalike:
            runner = self.runner
            req_index = self.req_index
            admission = self.make_chunk().admission
            def forward_scope(self):
                return contextlib.nullcontext()
        with self.assertRaises(LanePrivateError):
            admitted_chunk(Lookalike().forward_scope, runner=self.runner)

    def test_exception_restores_exact_owners(self):
        chunk = self.make_chunk()
        lane, operation = self.make_lane(chunk, bindings=self.bindings)
        lane.begin()
        self.attn.indexer.fail = True
        with self.assertRaises(RuntimeError):
            lane.layer(0)
        self.assertIs(self.attn.topk_indices_buffer, self.original)
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)
        self.assertIsNone(self.ws.active)
        self.assertFalse(self.bindings._active)
        # The bindings are reusable after the failed segment.
        self.attn.indexer.fail = False
        lane.layer(0)
        self.assertIs(self.attn.topk_indices_buffer, self.original)

    def test_stale_topk_owner_is_refused(self):
        replacement = Tensor((16, self.width), 'int32')
        self.attn.topk_indices_buffer = replacement
        with self.assertRaises(LanePrivateError):
            self.bindings.chunk_scope(self.make_chunk()).scope(0).__enter__()
        self.assertIs(self.attn.topk_indices_buffer, replacement)
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)
        self.assertIsNone(self.ws.active)
        self.assertFalse(self.bindings._active)

    def test_stale_uid_namespace_is_refused(self):
        other = Tensor((4,), 'int32')
        self.fc.QSA_REQ_UID = other
        with self.assertRaises(LanePrivateError):
            self.bindings.chunk_scope(self.make_chunk()).scope(0).__enter__()
        self.assertIs(self.fc.QSA_REQ_UID, other)
        self.assertIsNone(self.ws.active)

    def test_scratch_aliasing_decode_topk_is_refused(self):
        aliased = Tensor((self.chunk_tokens * self.width * 4,), 'uint8',
                         storage=self.original._storage)
        self.runtime._native_owned_buffers = (
            types.SimpleNamespace(name=lane_private.UID_BUFFER_NAME, tensor=self.private_uid),
            types.SimpleNamespace(name=lane_private.TOPK_SCRATCH_NAME, tensor=aliased))
        with self.assertRaises(LanePrivateError):
            self.discover()

    def test_capacity_and_alignment_are_enforced(self):
        small = Tensor((self.chunk_tokens * self.width * 4 - 1,), 'uint8')
        self.runtime._native_owned_buffers = (
            types.SimpleNamespace(name=lane_private.UID_BUFFER_NAME, tensor=self.private_uid),
            types.SimpleNamespace(name=lane_private.TOPK_SCRATCH_NAME, tensor=small))
        with self.assertRaises(LanePrivateError):
            self.discover()

        unaligned = Tensor((self.chunk_tokens * self.width * 4,), 'uint8',
                           storage=Storage(self.chunk_tokens * self.width * 4,
                                           base=Storage(16).data_ptr() + 4))
        self.runtime._native_owned_buffers = (
            types.SimpleNamespace(name=lane_private.UID_BUFFER_NAME, tensor=self.private_uid),
            types.SimpleNamespace(name=lane_private.TOPK_SCRATCH_NAME, tensor=unaligned))
        with self.assertRaises(LanePrivateError):
            self.discover()

        with self.assertRaises(LanePrivateError):
            self.bindings.chunk_scope(self.make_chunk(num_tokens=self.chunk_tokens + 1))

    def test_persistent_marlin_locks_private_shared_and_restored(self):
        stock = Tensor((82,), 'int32').fill_(91)
        kernel = types.SimpleNamespace(workspace=stock)
        self.attn.quant_method = types.SimpleNamespace(scheme=types.SimpleNamespace(
            linear_method=types.SimpleNamespace(kernel=kernel)))
        # A second host owner of the same storage preserves sharing privately.
        self.model.layers[0].workspace = stock
        bindings = self.discover()
        self.ws.manager.unlock()
        bindings.prepare_workspace()
        self.ws.manager.lock()
        self.assertEqual(bindings.audit()['marlin_lock_owners'], 2)
        self.assertEqual(bindings.audit()['marlin_lock_bytes'], 82*4)
        scope = bindings.chunk_scope(self.make_chunk())
        with scope.scope(0):
            private = kernel.workspace
            self.assertIs(self.model.layers[0].workspace, private)
            self.assertIsNot(private, stock)
            self.assertEqual(private.read(), 0)
            private.fill_(7)
        self.assertIs(kernel.workspace, stock)
        self.assertEqual(stock.read(), 91)
        with scope.scope(0):
            self.assertEqual(kernel.workspace.read(), 7)  # initialized only once
        kernel.workspace = Tensor((82,), 'int32')
        with self.assertRaisesRegex(LanePrivateError, 'owner changed'):
            with scope.scope(0):
                pass
        self.assertIs(self.model.layers[0].workspace, stock)
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)

    def test_actual_pinned_marlin_apply_uses_private_lock_during_prefill(self):
        path = ROOT / 'reference/vllm-src/model_executor/kernels/linear/mixed_precision/marlin.py'
        source = path.read_text()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
            'fef6ff0c4a945809bdc09c2bf684c0dceb7080d40f309fbc0535a94b992dc700')
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                   and n.name == 'MarlinLinearKernel')
        apply = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'apply_weights')
        calls = []
        def gemm(**kwargs):
            calls.append(kwargs['workspace'])
            kwargs['workspace'].fill_(7)
            return kwargs['input']
        namespace = dict(apply_gptq_marlin_linear=gemm)
        exec('from __future__ import annotations\n' + ast.get_source_segment(source, apply), namespace)
        stock = Tensor((82,), 'int32').fill_(91)
        kernel = types.SimpleNamespace(workspace=stock,
            config=types.SimpleNamespace(weight_type=1, partition_weight_shape=(4, 4), act_type=None),
            _get_weight_params=lambda layer: (None, None, None))
        method = namespace['apply_weights']
        self.attn.quant_method = types.SimpleNamespace(scheme=types.SimpleNamespace(
            linear_method=types.SimpleNamespace(kernel=kernel)))
        # Delivered J only changed generic workspace ownership: same lock was
        # passed to decode and prefill. The actual method shows the conflict.
        method(kernel, self.attn, 'decode')
        self.assertIs(calls[-1], stock)
        stock.fill_(91)
        bindings = self.discover()
        self.ws.manager.unlock()
        bindings.prepare_workspace()
        self.ws.manager.lock()
        with bindings.chunk_scope(self.make_chunk()).scope(0):
            method(kernel, self.attn, 'prefill')
            self.assertIsNot(calls[-1], stock)
            self.assertEqual(stock.read(), 91)
        self.assertIs(kernel.workspace, stock)
        method(kernel, self.attn, 'decode-again')
        self.assertIs(calls[-1], stock)

    def test_persistent_lock_geometry_and_partial_alias_refused(self):
        self.attn.quant_method = types.SimpleNamespace(workspace=Tensor((4,), 'bf16'))
        with self.assertRaisesRegex(LanePrivateError, 'int32 lock'):
            self.discover()
        stock = Tensor((82,), 'int32')
        self.attn.quant_method.workspace = stock
        self.model.layers[0].workspace = Tensor((81,), 'int32', storage=stock._storage, offset=4)
        with self.assertRaisesRegex(LanePrivateError, 'partially overlap'):
            self.discover()

    def test_capture_hook_funds_private_lane_before_split_capture(self):
        import graph_hooks
        from test_e3_graph_routing import make_router
        rig = make_router(capture=False)
        private = self.discover()
        self.ws.manager.unlock()
        self.ws.manager._current_workspaces[1] = None
        runner = types.SimpleNamespace(_flash_e3_graph_builder=lambda _: rig.router,
            _flash_s5_native=types.SimpleNamespace(private_bindings=private))
        capture = rig.router.capture_at_startup
        def checked_capture():
            self.assertTrue(private.audit()['workspace_prepared'])
            self.assertIsNotNone(self.ws.manager._current_workspaces[1])
            self.assertFalse(self.ws.manager._locked)
            return capture()
        with patch.object(rig.router, 'capture_at_startup', side_effect=checked_capture):
            with patch.dict(os.environ, {'FLASH_S5_OVERLAP': '1'}):
                graph_hooks.capture_banks(runner)
        self.ws.manager.lock()
        with self.ws.use_workspace_lane(1):
            self.assertIsNotNone(self.ws.manager._ensure_workspace_size(128))

    def test_actual_pinned_manager_locked_empty_lane_refuses_without_funding(self):
        module = FakeWorkspaceModule(2, self.torch)
        manager = module.manager
        stock = manager._ensure_workspace_size(128)
        manager.lock()
        with module.use_workspace_lane(1):
            with self.assertRaisesRegex(AssertionError, 'locked'):
                manager._ensure_workspace_size(128)
        self.assertIs(manager._current_workspaces[0], stock)
        self.assertIsNone(manager._current_workspaces[1])

    def test_funded_workspace_is_private_retained_and_locked_at_profile_capacity(self):
        manager = self.ws.manager
        slots = manager._current_workspaces
        self.assertFalse(lane_private._aliases(slots[0], slots[1]))
        self.assertEqual(self.bindings.prepare_workspace(), 128)
        self.assertEqual(self.bindings.audit()['workspace_bytes'], 128)
        with self.ws.use_workspace_lane(1):
            self.assertIs(manager._ensure_workspace_size(128), slots[1])
            with self.assertRaisesRegex(AssertionError, 'locked'):
                manager._ensure_workspace_size(129)
        manager._current_workspaces[1] = Tensor((128,), 'uint8')
        with self.assertRaisesRegex(LanePrivateError, 'retained funded'):
            with self.bindings.chunk_scope(self.make_chunk()).scope(0):
                pass

    def test_workspace_requires_profile_bound_before_lock_and_no_capture(self):
        other = self.discover()
        with self.assertRaisesRegex(LanePrivateError, 'before.*locks'):
            other.prepare_workspace()
        with self.assertRaisesRegex(LanePrivateError, 'retained funded'):
            with other.chunk_scope(self.make_chunk()).scope(0):
                pass
        self.ws.manager.unlock()
        self.runner.max_num_tokens = self.chunk_tokens - 1
        with self.assertRaisesRegex(LanePrivateError, 'profile token bound'):
            other.prepare_workspace()
        self.runner.max_num_tokens = self.chunk_tokens
        with patch.object(self.torch.cuda, 'is_current_stream_capturing', return_value=True):
            with self.assertRaisesRegex(LanePrivateError, 'non-capturing'):
                other.prepare_workspace()

    def test_workspace_lane_must_be_the_extra_lane(self):
        self.ws.lanes = 3
        with self.assertRaises(LanePrivateError):
            self.discover()
        self.ws.lanes = 2

    def test_absent_uid_state_is_refused(self):
        chunk = self.make_chunk()
        scope = self.bindings.chunk_scope(chunk)
        self.runner._qsa_uid_slots = None
        with self.assertRaises(LanePrivateError):
            scope.scope(0).__enter__()
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)
        self.assertIsNone(self.ws.active)
        self.runner._qsa_uid_slots = self.slots

        self.runner._qsa_uid_next = 0
        with self.assertRaises(LanePrivateError):
            self.bindings.chunk_scope(chunk).scope(0).__enter__()
        self.runner._qsa_uid_next = 7

    def test_admitted_chunk_is_required(self):
        chunk = self.make_chunk()
        self.assertIs(admitted_chunk(chunk.forward_scope, runner=self.runner), chunk)
        with self.assertRaises(LanePrivateError):
            admitted_chunk(lambda: None)
        other = types.SimpleNamespace(model=self.model)
        with self.assertRaises(LanePrivateError):
            admitted_chunk(chunk.forward_scope, runner=other)

    def test_worker_native_derives_the_chunk_scope(self):
        from native_startup import WorkerNative

        split = types.SimpleNamespace(prefill_stream=FakeStream(700),
                                      describe=lambda: 'split')
        bank = types.SimpleNamespace(measured_bytes=1024)
        comms = types.SimpleNamespace(
            tp=None, pp=None,
            describe=lambda: {'tp_world_size': 1, 'tp_warmed': False,
                              'basis': 'CPU private-scope fixture'})
        controller = types.SimpleNamespace(route='split', route_generation=0)
        native = WorkerNative(
            self.runner, split, bank, None, controller, comms, None, None,
            self.chunk_tokens, None, self.torch, private_bindings=self.bindings)
        chunk = self.make_chunk()
        lane = native.lane_forward(chunk.lane_inputs(),
                                   forward_context=chunk.forward_scope)
        self.assertIsNotNone(lane.private_scope)
        self.assertEqual(native.inventory()['private_bindings']['isolation'], 'partial')
        with lane._scopes(0):
            self.assertIs(self.attn.topk_indices_buffer, self.bindings.topk)
            self.assertIs(self.fc.QSA_REQ_UID, self.private_uid)
        self.assertIs(self.attn.topk_indices_buffer, self.original)
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)
        with self.assertRaises(LanePrivateError):
            native.lane_forward(chunk.lane_inputs(), forward_context=lambda: None)

    def test_ple_prefetch_needs_its_private_owner(self):
        embedding = types.SimpleNamespace(supports_prefetch=True)
        self.model.layers[0].ple = types.SimpleNamespace(ngram_embedding=embedding)
        with self.assertRaises(LanePrivateError):
            self.discover()

    def test_audit_names_unbound_owners(self):
        audit = self.bindings.audit()
        self.assertEqual(audit['isolation'], 'partial')
        self.assertEqual(audit['readiness'], 'unclaimed')
        self.assertEqual(audit['workspace_basis'], 'extra_last_lane')
        self.assertEqual(audit['workspace_lane'], 1)
        self.assertEqual(audit['workspace_lanes'], 2)
        layer = audit['qsa_layers'][0]
        self.assertEqual(layer['topk'], 'private_view')
        self.assertEqual(layer['topk_width'], self.width)
        self.assertEqual(layer['rowcache'], 'absent')
        self.assertIn('kv_cache', layer['history'])
        self.assertTrue(audit['unbound'])
        self.assertTrue(any('row cache' in item for item in audit['unbound']))
        self.assertTrue(any('request rows' in item for item in audit['unbound']))

    def test_private_scope_enters_without_a_qsa_layer(self):
        # Prefix/suffix segments own the UID and workspace even with no layer.
        lane = LaneForward(self.model, self.make_chunk().lane_inputs(),
                           stream=FakeStream(700),
                           forward_context=self.make_chunk().forward_scope,
                           torch_module=self.torch, clock=lambda: 0,
                           private_scope=self.bindings.chunk_scope(
                               self.make_chunk()).scope)
        with lane._scopes(None):
            self.assertIs(self.fc.QSA_REQ_UID, self.private_uid)
            self.assertEqual(self.private_uid.read(0), 7)
            self.assertEqual(self.ws.active, 1)
            self.assertIs(self.attn.topk_indices_buffer, self.original)
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)
        self.assertIsNone(self.ws.active)


if __name__ == '__main__':
    unittest.main()
