"""Native QSA submission credit: accepted operations bounded by real submissions.

The concern under test is the actual source path, not a controller receipt:

* ``QsaLayerDriver._bucket`` returned ``None`` both for a typed ``E3Pending``
  and for the sparse selection sentinel.  ``LaneForward`` turned that into a
  paused segment, ``LaneDispatcher`` recorded an event and accepted a new
  ``op_seq`` for it, and the engine's ``OpSeqTracker`` consumed that sequence.
  A delayed consumer or selection readback therefore grew accepted operation
  metadata (journal ``completed_op_seqs``/``proofs``) with polling ticks while
  no GPU work was submitted.
* ``QsaPrefillOperation.submission_credit`` is the additive, host-query-only
  seam over the same monotonic evidence ``advance`` consults.  The dispatcher
  queries it *before* its partial-enqueue boundary and returns typed pending
  for the unchanged command.

These tests drive the real ``QsaLayerDriver``/``LaneForward``/``LaneDispatcher``
/``NativeWorkerBridge``/``OpSeqTracker`` over the generated dense and sparse
attention arithmetic with a controlled fake device.  More than a thousand
delayed polls must leave the accepted op list, event lists, copies, kernels and
lane records exactly where the real submissions left them; the eventual command
must run the same QSA output path; the legacy (no-seam) source shape must still
reproduce the metadata growth and the real 65-element ``_bounded_value``
refusal, whose legitimate fixed-model maximum is proven below.
"""
import ast
import contextlib
import json
import os
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
sys.path.insert(0, str(ROOT / 'harness'))

import engine_driver as ed  # noqa: E402
import e3_types as et  # noqa: E402
import qsa_driver  # noqa: E402
import qsa_operation as qo  # noqa: E402
import sparse_history_dma as shd  # noqa: E402
import wire_v1 as w  # noqa: E402
import worker_bridge  # noqa: E402
from lane_dispatch import (  # noqa: E402
    LaneDispatchError,
    LaneDispatcher,
    SegmentCommand as LocalCommand,
)
from model_hooks import LaneForward  # noqa: E402
import test_e3_sparse_history_dma as sparse_test  # noqa: E402

REFERENCE = ROOT / 'reference/vllm-src/models/qwen4_exp/nvidia'
MODEL_CONFIG = ROOT / 'reference/model-config.json'

LAYER = 'qsa.0'
SIDE = 'qsa_side'
ROW_WIDTH = 512
HEAD_DIM = ROW_WIDTH // 2
QUERY_HEADS = 6
NUM_KV_HEADS = 1
PAGE_TOKENS = shd.PAGE_ROWS
DENSE_TOKENS = 8
# 25 dense pages: the first count above the 24-page arena, so the generated
# sparse helper is actually eligible.
SPARSE_TOKENS = 24 * PAGE_TOKENS + 1
SPARSE_PAGES = 25
POLLS = 1100


# --- controlled device / stream fakes ---------------------------------------

class ControllableEvent:
    """A CUDA event whose host query the test decides; it never waits."""

    def __init__(self, bus):
        self.bus = bus
        self.stream = None
        self.complete = False
        bus.events.append(self)

    def record(self, stream=None):
        self.stream = stream

    def query(self):
        return self.complete

    def synchronize(self):
        raise AssertionError('the lane and operation must never synchronize')


class EventBus:
    def __init__(self):
        self.events = []

    def event(self):
        return ControllableEvent(self)

    def complete_all(self):
        for event in self.events:
            event.complete = True


def bus_torch(bus):
    torch = types.SimpleNamespace()
    torch.cuda = types.SimpleNamespace(
        Event=lambda enable_timing=False: bus.event(),
        stream=lambda stream: contextlib.nullcontext(stream))
    torch.inference_mode = lambda: contextlib.nullcontext()
    return torch


class FakeStream:
    def __init__(self, token):
        self.cuda_stream = token
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)

    def synchronize(self):
        raise AssertionError('the lane must never synchronize a prefill stream')


class OutputTensor(sparse_test.T):
    """The attention output storage; ``zero_`` is the only extra pinned op."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zeroed = 0

    def zero_(self):
        self.zeroed += 1
        return self


class LaneContext:
    """The chunk forward context: a context manager carrying the metadata."""

    def __init__(self, metadata):
        self.attn_metadata = metadata

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- fake model / attention -------------------------------------------------

class FakeIndexer:
    def __init__(self, buffer):
        self._buffer = buffer
        self.raw_key_cache = types.SimpleNamespace(prefix=SIDE)
        self.packed_output_width = buffer.shape[1]

    def __call__(self, projected_qk, positions, logical_indices):
        return sparse_test.T(tuple(logical_indices.shape), device='cuda:0',
                             dtype='torch.int64')


class FakeImpl:
    def __init__(self):
        self.updates = 0

    def do_kv_cache_update(self, attn, key, value, kv_cache, slot_mapping):
        self.updates += 1


class FakeAttention:
    """The pinned QSA module surface ``QsaLayerDriver`` composes against."""

    def __init__(self, stack, kv_cache):
        self.stack = stack
        self.layer_name = LAYER
        self.kv_cache = kv_cache
        self._qsa_host_kv = stack.source
        # Distinct selected tokens in the first value column, padding in the
        # second: the sparse union sees exactly 32 sorted, unique tokens.
        values = []
        for row in range(stack.tokens):
            values.extend((row % 32, -1))
        self.topk_indices_buffer = sparse_test.T(
            (stack.tokens, 2), device='cuda:0', dtype='torch.int32',
            values=values)
        self.indexer = FakeIndexer(self.topk_indices_buffer)
        self.impl = FakeImpl()
        self._max_decode_query_len = 1
        # The generated helper's own dense-transport preconditions.
        self._qsa_kv_offload = True
        self._qsa_stage_min_rows = 1
        self._qsa_stage_arena_bytes = shd.ARENA_ROWS * shd.ROW_BYTES
        self.pre_calls = 0
        self.post_calls = 0

    def _flash_e3_attention_pre(self, positions, hidden_states):
        self.pre_calls += 1
        tokens = self.stack.tokens
        return {
            'query': sparse_test.T((tokens, QUERY_HEADS, HEAD_DIM),
                                   device='cuda:0', dtype='torch.bfloat16'),
            'key': sparse_test.T((tokens, NUM_KV_HEADS, HEAD_DIM),
                                 device='cuda:0', dtype='torch.bfloat16'),
            'value': sparse_test.T((tokens, NUM_KV_HEADS, HEAD_DIM),
                                   device='cuda:0', dtype='torch.bfloat16'),
            'gate': None,
            'attn_output': self.stack.output,
            'projected_qk': sparse_test.T((tokens, 4), device='cuda:0',
                                          dtype='torch.bfloat16'),
            'num_tokens': tokens,
        }

    def _flash_e3_attention_post(self, state):
        self.post_calls += 1
        return state['attn_output']


class FakeQsaLayer:
    layer_type = 'full_attention'

    def __init__(self, attention, tokens):
        self.self_attn = attention
        self.tokens = tokens

    def _flash_e3_layer_pre(self, hidden, block_output, injection, *,
                            input_ids, query_start_loc, ngram_context):
        return hidden, block_output, injection

    def _flash_e3_layer_post(self, hidden, attn_out, injection):
        return (sparse_test.T((self.tokens, 16), device='cuda:0',
                              dtype='torch.bfloat16'), None, None)


class FakeGdnLayer:
    layer_type = 'linear_attention'


class FakeModel:
    """Only the generated lane-forward entry points the lane calls."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.layers = [None, FakeGdnLayer()]
        self.start_layer = 0
        self.end_layer = len(self.layers)
        self._mtp_hidden_buffer = None
        self.forward_layer_calls = []

    def _flash_e3_forward_prefix(self, input_ids, intermediate_tensors,
                                 inputs_embeds=None):
        return sparse_test.T((self.tokens, 16), device='cuda:0',
                             dtype='torch.bfloat16')

    def _flash_e3_forward_layer(self, layer_idx, hidden, block_output,
                                injection, positions, input_ids,
                                query_start_loc, ngram_context):
        self.forward_layer_calls.append(layer_idx)
        return (sparse_test.T((self.tokens, 16), device='cuda:0',
                              dtype='torch.bfloat16'), None, None)

    def _flash_e3_forward_suffix(self, hidden, block_output, injection, last):
        return {'hidden_states': hidden}

    def _flash_e3_is_last_stage(self):
        return True


class CreditChunk:
    def __init__(self, stack, admission):
        self.stack = stack
        self.admission = admission
        self.req_index = stack.rank
        self.finished = 0

    def lane_inputs(self):
        return dict(self.stack.inputs)

    def forward_scope(self):
        return LaneContext(self.stack.attn_metadata)

    def finish_state(self):
        self.finished += 1
        return True


class CreditFactory:
    def __init__(self, stack):
        self.stack = stack
        self.chunks = []
        self.released = 0

    def admit(self, admission):
        chunk = CreditChunk(self.stack, admission)
        self.chunks.append(chunk)
        return chunk

    def release(self):
        self.released += 1
        return True


class CreditBinding:
    """The worker-local source binding the bridge proves and then closes."""

    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.transaction = descriptor.transaction
        self.state_epoch = descriptor.state_epoch
        self.holds = {}
        self.closed = 0

    def close(self, registry):
        self.closed += 1
        return True


class CreditServer:
    """The handoff server surface: open/begin/end/abort/clear + operation."""

    def __init__(self, stack):
        self.stack = stack
        self.operation = None
        self.handoff = None
        self.calls = []
        self.runtime = types.SimpleNamespace(
            state=types.SimpleNamespace(worker_rank=stack.rank),
            release=lambda transaction: True)

    def attach_worker(self, worker):
        self.worker = worker
        return self

    def open(self, message):
        self.handoff = message.transaction
        self.operation = self.stack.make_operation(message.transaction)
        self.stack.operation = self.operation
        self.calls.append(('open', message.transaction))
        return types.SimpleNamespace(transaction=message.transaction,
                                     accepted=True, lane=self.stack.lane)

    def begin_prefill_operation(self, runner, marker):
        operation = self.operation
        operation.register_source(LAYER, self.stack.source,
                                  versions=self.stack.versions)
        operation.begin_forward(layers=(LAYER,))
        self.calls.append(('begin', marker.flash_e3_transaction))

    def end_prefill_operation(self, runner):
        self.operation.end_forward()
        self.calls.append(('end',))

    def abort_prefill_operation(self, runner):
        self.operation.cancel('abort')
        self.calls.append(('abort',))

    def clear_native(self, transaction):
        if self.handoff is not None and self.handoff != transaction:
            raise AssertionError('clear_native for another transaction')
        self.calls.append(('clear', transaction))
        return True


class LegacyOperation:
    """The real operation without the additive host-credit seam.

    This is the pre-change source shape: ``QsaLayerDriver`` finds no
    ``submission_credit`` and keeps its old paused behaviour.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name == 'submission_credit':
            raise AttributeError(name)
        return getattr(self._inner, name)


class OracleTrap:
    """Records every row1 qualification; a credit query must never run one."""

    def __init__(self):
        self.calls = []

    def bucket(self, run, bucket, arena):
        self.calls.append('bucket')

    def check_releasable(self, run, *, cancelled):
        self.calls.append('check_releasable')

    def finish_layer(self, run, *, hold_stats):
        self.calls.append('finish_layer')


class Stack:
    """One rank's real driver/lane/dispatcher/bridge over controlled fakes."""

    def __init__(self, harness, *, rank, route, transaction, ranks,
                 legacy=False, driverless=False):
        self.h = harness
        self.rank = rank
        self.route = route
        self.transaction = transaction
        self.ranks = tuple(ranks)
        self.legacy = legacy
        self.driverless = driverless
        self.tokens = DENSE_TOKENS if route == 'dense' else SPARSE_TOKENS
        self.mapping = ((0,) if route == 'dense'
                        else tuple(range(SPARSE_PAGES)))
        self.versions = tuple(range(1, len(self.mapping) + 3))
        self.stage = 0
        self.lane = et.LaneId(rank, et.LaneRole.PREFILL, 3, 0)

        self.device = sparse_test.FakeSparseDevice(auto_ready=False)
        self.bus = EventBus()
        self.torch = bus_torch(self.bus)
        self.source = sparse_test.T(
            (len(self.mapping), NUM_KV_HEADS, PAGE_TOKENS, ROW_WIDTH),
            device='cpu', dtype='torch.bfloat16', pinned=True)
        self.arena = sparse_test.T(
            (shd.ARENA_PAGES, NUM_KV_HEADS, PAGE_TOKENS, ROW_WIDTH),
            device='cuda:0', dtype='torch.bfloat16',
            storage=sparse_test.Rows(shd.ARENA_ROWS))
        self.arena.address = 0x900000 + 0x100000 * rank
        self.table = sparse_test.T((260,), device='cuda:0',
                                   dtype='torch.int64', values=[0] * 260)
        self.block_table = self.table.reshape(1, 260)
        self.kv_cache = sparse_test.T(
            (260, NUM_KV_HEADS, PAGE_TOKENS, 2 * HEAD_DIM),
            device='cuda:0', dtype='torch.bfloat16')
        self.output = OutputTensor((self.tokens, QUERY_HEADS, HEAD_DIM),
                                   device='cuda:0', dtype='torch.bfloat16')
        harness.arenas[id(self.kv_cache)] = self.arena

        self.side = types.SimpleNamespace(num_actual_tokens=self.tokens)
        self.main = types.SimpleNamespace(
            num_actual_tokens=self.tokens,
            slot_mapping=sparse_test.T((self.tokens,), device='cuda:0',
                                       dtype='torch.int64',
                                       values=list(range(self.tokens))),
            max_query_len=self.tokens,
            _qsa_host_meta=((0, self.tokens), (self.tokens,)),
            query_start_loc=sparse_test.T((2,), device='cpu',
                                          dtype='torch.int32',
                                          values=[0, self.tokens]),
            seq_lens=sparse_test.T((1,), device='cpu', dtype='torch.int32',
                                   values=[self.tokens]),
            block_table=self.block_table)
        self.attn_metadata = {LAYER: self.main, SIDE: self.side}
        self.inputs = {
            'positions': sparse_test.T((self.tokens,), device='cuda:0',
                                       dtype='torch.int64',
                                       values=list(range(self.tokens))),
            'input_ids': sparse_test.T((self.tokens,), device='cuda:0',
                                       dtype='torch.int64',
                                       values=[1] * self.tokens),
            'query_start_loc': sparse_test.T((2,), device='cpu',
                                             dtype='torch.int32',
                                             values=[0, self.tokens]),
            'ngram_context': sparse_test.T((0,), device='cuda:0',
                                           dtype='torch.int64', values=[]),
        }

        self.model = FakeModel(self.tokens)
        self.attention = FakeAttention(self, self.kv_cache)
        self.model.layers[0] = FakeQsaLayer(self.attention, self.tokens)
        self.holds = []

        self.operation = None
        self.driver = None
        self.split = types.SimpleNamespace(
            prefill_stream=FakeStream('prefill-%d' % rank))
        self.native = types.SimpleNamespace(
            split=self.split,
            comms=types.SimpleNamespace(pp=None),
            copy_stream=FakeStream('copy-%d' % rank),
            lane_forward=self._lane_forward)
        self.server = None if driverless else CreditServer(self)
        self.factory = CreditFactory(self)
        self.dispatcher = LaneDispatcher(
            'runner-%d' % rank, self.native, self.factory, server=self.server,
            driver_factory=None if driverless else self._driver_factory,
            torch_module=self.torch, clock=lambda: 1)
        self.bridge = None if driverless else worker_bridge.NativeWorkerBridge(
            self.dispatcher, server=self.server,
            source_binder=self._source_binder)

    # -- construction hooks -------------------------------------------------
    def make_operation(self, transaction):
        provider = qo.metadata_plan_provider(
            sparse_test.Snapshot([sparse_test.Row()], tuple(range(260))),
            layer_tables=((LAYER, self.mapping),))
        operation = qo.QsaPrefillOperation(
            transaction, self.lane, device=self.device, plan_provider=provider)
        operation.state_epoch = 1
        return operation

    def _source_binder(self, dispatcher, admission, descriptor, rank=None):
        descriptor.validate_frame(admission, (self.rank,))
        return CreditBinding(descriptor)

    def _forward_scope(self):
        return self.dispatcher.chunk.forward_scope()

    def _driver_factory(self, operation, module, context):
        self.operation = operation
        kwargs = {}
        if self.route == 'sparse':
            hold = shd.SparseSourceHold(
                sparse_test.make_owner(self.lane, LAYER, self.mapping,
                                       self.tokens,
                                       transaction=operation.transaction),
                allocator=sparse_test.FakeAllocator())
            self.holds.append(hold)
            kwargs['sparse_staged_operation'] = self.h.helper_sparse
            kwargs['source_holds'] = {LAYER: hold}
        inner = LegacyOperation(operation) if self.legacy else operation
        self.driver = qsa_driver.QsaLayerDriver(
            inner, staged_operation=self.h.helper_dense,
            forward_context=self._forward_scope, torch_module=self.torch,
            clock=lambda: 0, **kwargs)
        return self.driver

    def _lane_forward(self, inputs, forward_context=None, attention_driver=None):
        return LaneForward(self.model, inputs,
                           stream=self.split.prefill_stream,
                           forward_context=forward_context,
                           attention_driver=attention_driver,
                           torch_module=self.torch, clock=lambda: 0)

    # -- observation --------------------------------------------------------
    def consumer_event(self):
        events = [event for event in self.device.events
                  if event.token == self.device.compute and not event.ready]
        if not events:
            raise AssertionError('no pending attention consumer event')
        return events[-1]

    def ready_attention(self):
        self.consumer_event().ready = True

    def ready_selection(self):
        copies = [event for event in self.device.events
                  if event.token == self.device.copy and not event.ready]
        if not copies:
            raise AssertionError('no pending selection readback event')
        copies[-1].ready = True

    def complete_all(self):
        self.device.ready_all()
        self.bus.complete_all()


class World:
    """All ranks of one transaction with per-rank ``OpSeqTracker`` state."""

    def __init__(self, harness, *, route='dense', ranks=(0,), txn_seq=1,
                 legacy=False, driverless=False):
        self.h = harness
        self.route = route
        self.ranks = tuple(ranks)
        self.tokens = DENSE_TOKENS if route == 'dense' else SPARSE_TOKENS
        self.txn = et.TransactionId(engine_epoch=1,
                                    request_uid='credit-' + route,
                                    request_generation=0, chunk_start=0,
                                    chunk_end=self.tokens, txn_seq=txn_seq)
        self.admission = w.AdmissionRecord(
            transaction=self.txn, prompt_end=self.tokens + 1, state_epoch=1,
            committed_end=0, block_tables=((1,),),
            slot_mappings=(tuple(range(self.tokens)),),
            new_block_ids_to_zero=(1,), cow_pairs=(),
            input_ids=tuple(index % 7 for index in range(self.tokens)))
        self.lease = w.SourceLeaseDescriptor(self.txn, 1, (1,), self.ranks)
        self.stacks = tuple(
            Stack(harness, rank=rank, route=route, transaction=self.txn,
                  ranks=self.ranks, legacy=legacy, driverless=driverless)
            for rank in self.ranks)
        self.by_rank = {stack.rank: stack for stack in self.stacks}
        self.trackers = {rank: w.OpSeqTracker() for rank in self.ranks}
        self.accepted = {rank: [] for rank in self.ranks}
        self.kinds = {rank: {} for rank in self.ranks}
        self.commands = []

    # -- driving ------------------------------------------------------------
    def open(self):
        if self.stacks[0].driverless:
            chunk = worker_bridge.admission_to_chunk(self.admission)
            for stack in self.stacks:
                stack.operation = stack.make_operation(self.txn)
                stack.dispatcher.open(self.txn, chunk, stack.operation)
            return {}
        return {stack.rank: stack.bridge.open(
                    w.OpenRequest(self.admission, self.ranks, 1, 8))
                for stack in self.stacks}

    def descriptor(self, kind, layer):
        return w.SegmentDescriptor(
            self.txn, w.SegmentKind(kind), self.stacks[0].stage,
            layer_start=layer,
            layer_end=None if layer is None else layer + 1)

    def send(self, kind, layer=None, source=None):
        descriptor = self.descriptor(kind, layer)
        # A pending proposal is never replaced: every rank retries the same
        # descriptor and sequence.
        proposals = {rank: self.trackers[rank].propose(descriptor)
                     for rank in self.ranks}
        if len(set(proposals.values())) != 1:
            raise AssertionError('ranks proposed different op_seq values')
        command = w.SegmentCommand(descriptor, proposals[self.ranks[0]],
                                   self.ranks, source=source)
        self.commands.append(command)
        return command, self.deliver(command)

    def deliver(self, command):
        acks = {}
        for rank in self.ranks:
            stack = self.by_rank[rank]
            if stack.bridge is None:
                try:
                    result = stack.dispatcher.submit(command)
                except et.E3Pending:
                    ack = types.SimpleNamespace(
                        accepted=False, pending=True, skipped=False,
                        op_seq=command.op_seq, paused=False)
                else:
                    ack = types.SimpleNamespace(
                        accepted=True, pending=False, skipped=False,
                        op_seq=command.op_seq,
                        paused=bool(result['result'].get('paused')))
            else:
                ack = stack.bridge.submit(command)
            acks[rank] = ack
            tracker = self.trackers[rank]
            if ack.accepted:
                # Only the first acceptance spends this sequence; an identical
                # replay is idempotent and never adds an accepted operation.
                if tracker.accepted(command.op_seq, command.descriptor):
                    self.accepted[rank].append(command.op_seq)
                    self.kinds[rank][command.op_seq] = command.kind.value
            else:
                if not ack.pending:
                    raise AssertionError('submission neither accepted nor pending')
                if ack.op_seq != command.op_seq:
                    raise AssertionError('pending did not keep the op_seq')
                if tracker.pending != (command.op_seq, command.descriptor):
                    raise AssertionError('pending proposal was not retained')
        return acks

    def complete_all(self):
        for stack in self.stacks:
            stack.complete_all()


class CreditHarness:
    """Generated dense+sparse code, controlled device and real source stack."""

    @classmethod
    def setUpClass(cls):
        if not REFERENCE.is_dir():
            raise unittest.SkipTest(
                'reference/vllm-src is not available on this host')
        import arena_attention
        import operation_source
        import runner_hooks
        ops = arena_attention.generate((REFERENCE / 'ops/qsa.py').read_text())
        cls.generated_ops = operation_source.generate_ops(ops)
        qsa = runner_hooks.qsa_staged((REFERENCE / 'qsa.py').read_text())
        cls.generated_qsa = operation_source.generate_layer(qsa)

    def setUp(self):
        self.torch = sparse_test.FakeTorch()
        self.kernel = sparse_test.FakeKernel('gqa')
        self.union_kernels = sparse_test.union_kernels()
        self.compile_generated()
        self.arenas = {}
        self.install_package()
        self.worlds = []

    # -- generated code -----------------------------------------------------
    def compile_generated(self):
        ops_module = ast.Module(body=[
            sparse_test._function_node(self.generated_ops,
                                       'flash_s5_arena_attention'),
            sparse_test._function_node(self.generated_ops,
                                       'flash_s5_sparse_arena_attention'),
            sparse_test._function_node(self.generated_ops, '_s6_union'),
        ], type_ignores=[])
        ast.fix_missing_locations(ops_module)
        ops_globals = {
            '__name__': 'ops.qsa',
            'torch': self.torch,
            'os': os,
            'triton': sparse_test.FakeTriton(),
            '_STAGE_TIMER': None,
            '_qsa_sparse_staged_gqa_kernel': self.kernel,
            '_qsa_gather_selected_host_rows': sparse_test.FakeKernel('gather'),
            '_select_config': lambda rows, heads, prefill, width: (
                64, 4, 1, None),
        }
        ops_globals.update(self.union_kernels)
        exec(compile(ops_module, '<generated ops>', 'exec'), ops_globals)
        helper_module = ast.Module(body=[
            sparse_test._function_node(self.generated_qsa,
                                       '_flash_s5_staged_operation'),
            sparse_test._function_node(self.generated_qsa,
                                       '_flash_s5_sparse_staged_operation'),
        ], type_ignores=[])
        ast.fix_missing_locations(helper_module)
        helper_globals = {'__name__': 'models.qwen4_exp.nvidia.qsa',
                          '__package__': 'models.qwen4_exp.nvidia',
                          'torch': self.torch}
        exec(compile(helper_module, '<generated qsa>', 'exec'),
             helper_globals)
        self.attention_dense = ops_globals['flash_s5_arena_attention']
        self.attention_sparse = ops_globals['flash_s5_sparse_arena_attention']
        self.helper_dense = helper_globals['_flash_s5_staged_operation']
        self.helper_sparse = helper_globals['_flash_s5_sparse_staged_operation']

    def install_package(self):
        package_name = 'models.qwen4_exp.nvidia.ops.qsa'
        package = sparse_test._module(
            package_name,
            flash_s5_sparse_arena_attention=self.attention_sparse,
            flash_s5_arena_attention=self.attention_dense,
            flash_e2_arena_attention=lambda *args, **kwargs: None,
            qsa_get_staging_arena=self.arena_for)
        for name in ('models', 'models.qwen4_exp', 'models.qwen4_exp.nvidia',
                     'models.qwen4_exp.nvidia.ops'):
            sys.modules.setdefault(name, sparse_test._module(name))
        sys.modules[package_name] = package
        self.addCleanup(sys.modules.pop, package_name, None)

    def arena_for(self, kv_cache, nbytes):
        arena = self.arenas.get(id(kv_cache))
        if arena is None:
            raise AssertionError('no staging arena registered for this kv cache')
        return arena

    # -- world helpers ------------------------------------------------------
    def world(self, *, route='dense', ranks=(0,), txn_seq=1, legacy=False,
              driverless=False):
        world = World(self, route=route, ranks=ranks, txn_seq=txn_seq,
                      legacy=legacy, driverless=driverless)
        self.worlds.append(world)
        world.open()
        return world

    def stack(self, world, rank=0):
        return world.by_rank[rank]

    def expect_accepted(self, world, command, acks, *, paused=None):
        for rank in world.ranks:
            ack = acks[rank]
            self.assertTrue(ack.accepted, ack)
            self.assertFalse(ack.pending)
            self.assertEqual(ack.op_seq, command.op_seq)
            if paused is not None:
                self.assertEqual(ack.paused, paused)

    def expect_pending(self, world, command, acks):
        for rank in world.ranks:
            ack = acks[rank]
            self.assertTrue(ack.pending, ack)
            self.assertFalse(ack.accepted)
            self.assertFalse(ack.skipped)
            self.assertEqual(ack.op_seq, command.op_seq)
            self.assertEqual(world.trackers[rank].pending,
                             (command.op_seq, command.descriptor))

    def do(self, world, kind, *, layer=None, source=None, paused=None):
        command, acks = world.send(kind, layer=layer, source=source)
        self.expect_accepted(world, command, acks, paused=paused)
        return command, acks

    def poll(self, world, command, count):
        for _ in range(count):
            self.expect_pending(world, command, world.deliver(command))

    def submission_state(self, world):
        """Everything a pure pending attempt must leave untouched."""
        total = dict(device_events=0, page_copies=0, row_copies=0,
                     index_copies=0, host_reads=0, waits=0, synchronized=0,
                     kernels=0, records=0, pending=0, copy_events=0,
                     consumer_events=0, selection_copies=0, lane_segments=0,
                     dispatcher_events=0, bus_events=0)
        # The generated helpers of every stack share this one fake kernel.
        total['kernels'] = len(self.kernel.calls)
        for stack in world.stacks:
            device = stack.device
            operation = stack.operation
            total['device_events'] += len(device.events)
            total['page_copies'] += len(device.page_copies)
            total['row_copies'] += len(device.copies)
            total['index_copies'] += len(device.index_copies)
            total['host_reads'] += len(device.host_reads)
            total['waits'] += len(device.waits)
            total['synchronized'] += device.synchronized
            total['records'] += len(operation._records)
            total['pending'] += operation.pending
            for key in ('copy_events', 'consumer_events', 'selection_copies'):
                total[key] += operation.stats[key]
            lane = stack.dispatcher.lane
            total['lane_segments'] += 0 if lane is None else len(lane.segments)
            total['dispatcher_events'] += len(stack.dispatcher.events)
            total['bus_events'] += len(stack.bus.events)
        return total

    def begin_layer0(self, world, *, first_paused=True):
        """Drive prepare/prefix/pre and the first real bucket submission."""
        self.do(world, 'prepare', source=world.lease)
        self.do(world, 'prefix', paused=False)
        self.do(world, 'layer', layer=0, paused=True)
        self.do(world, 'layer', layer=0, paused=first_paused)

    def finish(self, world):
        """The GDN layer, suffix and state finalize after layer 0's tail."""
        self.do(world, 'layer', layer=1, paused=False)
        self.do(world, 'suffix', paused=False)
        self.do(world, 'state_finalize', paused=False)

    def complete_and_release(self, world):
        world.complete_all()
        reports = {}
        for rank in world.ranks:
            reports[rank] = world.by_rank[rank].bridge.poll(
                w.PollRequest(world.txn, 1, 8))
        for rank in world.ranks:
            stack = world.by_rank[rank]
            ack = stack.bridge.release(w.ReleaseRequest(
                world.txn, 1, ((rank, reports[rank].completed_sequence),),
                True))
            self.assertTrue(ack.released, ack)
            self.assertTrue(ack.request_closed, ack)
        return reports


# --- dense attention credit -------------------------------------------------

class AttentionCreditTests(CreditHarness, unittest.TestCase):
    def test_delayed_attention_consumer_does_not_consume_accepted_ops(self):
        world = self.world(route='dense')
        stack = self.stack(world)
        self.begin_layer0(world)
        self.assertEqual(world.accepted[0], [0, 1, 2, 3])
        # Every accepted operation is a real submission: the producer, the
        # bucket copy and the attention consumer each keep their own ticket.
        self.assertEqual(stack.operation.stats['producers'], 1)
        self.assertEqual(stack.operation.stats['buckets'], 1)
        self.assertEqual(stack.operation.stats['copy_events'], 1)
        self.assertEqual(stack.operation.stats['consumer_events'], 1)
        stages = [record.stage for record in stack.operation._records]
        self.assertEqual(stages, [qo.Stage.PRODUCER, qo.Stage.HISTORY_READY,
                                  qo.Stage.ATTENTION])
        self.assertTrue(all(record.event is not None
                            for record in stack.operation._records))
        self.assertEqual(len({record.ticket.op_seq
                              for record in stack.operation._records}), 3)
        self.assertEqual(stack.attention.pre_calls, 1)
        self.assertEqual(stack.attention.post_calls, 0)
        self.assertEqual(stack.dispatcher.lane.segments[-1].kind,
                         'layer-paused')

        command, acks = world.send('layer', layer=0)
        self.expect_pending(world, command, acks)
        before = self.submission_state(world)
        self.poll(world, command, POLLS)
        self.assertEqual(before, self.submission_state(world))
        self.assertEqual(world.accepted[0], [0, 1, 2, 3])
        self.assertEqual(world.trackers[0].pending,
                         (command.op_seq, command.descriptor))
        self.assertEqual(world.trackers[0].replays, 0)
        self.assertEqual(len(stack.dispatcher.events), 4)
        self.assertEqual(stack.operation.stats['buckets'], 1)
        self.assertEqual(stack.operation.pending, 3)

        # Only the queried consumer event, not elapsed time or a flag, grants
        # the next submission.  It is still the same command and op_seq.
        stack.ready_attention()
        acks = world.deliver(command)
        self.expect_accepted(world, command, acks, paused=False)
        self.assertEqual(world.accepted[0], [0, 1, 2, 3, 4])
        self.assertEqual(stack.attention.post_calls, 1)
        self.assertEqual(stack.attention.pre_calls, 1)
        self.assertEqual(stack.operation.stats['buckets'], 1)
        # The eventual command ran the same generated QSA output path: the
        # staged kernel was launched against the operation's own output tensor,
        # which the tail then consumed through the additive post method.
        self.assertEqual(stack.output.zeroed, 1)
        self.assertTrue(any(getattr(arg, 'address', None) == stack.output.address
                            and getattr(arg, 'shape', None)
                            == stack.output.shape
                            for _, args, kwargs in self.kernel.calls
                            for arg in list(args) + list(kwargs.values())),
                        'the generated attention kernel did not write into the '
                        'operation output tensor')

        self.finish(world)
        self.assertEqual(world.accepted[0], [0, 1, 2, 3, 4, 5, 6, 7])
        report = self.complete_and_release(world)[0]
        self.assertEqual(report.completed_op_seqs, tuple(world.accepted[0]))
        kinds = [proof.kind for proof in report.proofs]
        self.assertEqual(kinds.count(et.CompletionKind.GPU_EVENT),
                         len(world.accepted[0]))
        self.assertEqual(kinds.count(et.CompletionKind.STATE_WRITE), 1)
        self.assertEqual(len(report.proofs), len(world.accepted[0]) + 1)
        self.assertEqual(len(report.host_submit_ns), len(world.accepted[0]))
        self.assertEqual(stack.device.synchronized, 0)

    def test_credit_reasons_are_the_source_predicates(self):
        world = self.world(route='dense')
        stack = self.stack(world)
        self.begin_layer0(world)
        command, acks = world.send('layer', layer=0)
        self.expect_pending(world, command, acks)
        operation = stack.operation
        credit = operation.submission_credit()
        self.assertFalse(credit.can_submit)
        self.assertEqual(credit.reason, 'attention consumer not observed')
        self.assertEqual(credit.pending_buckets, 1)
        self.assertEqual(credit.pending_selection, 0)

        # The producer/store event completing is not attention credit.
        producer = stack.device.events[0]
        self.assertEqual(producer.token, stack.device.compute)
        producer.ready = True
        credit = stack.driver.submission_credit(0)
        self.assertFalse(credit.can_submit)
        self.assertEqual(credit.reason, 'attention consumer not observed')

        stack.ready_attention()
        credit = stack.driver.submission_credit(0)
        self.assertTrue(credit.can_submit)
        self.assertEqual((credit.pending_buckets, credit.pending_selection),
                         (0, 0))

        # No phase is outstanding on another layer yet.
        self.assertIsNone(stack.driver.submission_credit(1))

    def test_credit_query_never_runs_the_row1_oracle(self):
        world = self.world(route='dense')
        stack = self.stack(world)
        self.do(world, 'prepare', source=world.lease)
        self.do(world, 'prefix', paused=False)
        trap = OracleTrap()
        stack.operation.qualification = trap
        self.do(world, 'layer', layer=0, paused=True)
        self.do(world, 'layer', layer=0, paused=True)
        command, acks = world.send('layer', layer=0)
        self.expect_pending(world, command, acks)
        self.poll(world, command, 300)
        self.assertEqual(trap.calls, [])
        stack.ready_attention()
        acks = world.deliver(command)
        self.expect_accepted(world, command, acks, paused=False)
        # The oracle runs only in advance(), once the queried consumer exists.
        self.assertEqual(trap.calls, ['bucket'])

    def test_partial_enqueue_failure_still_quarantines(self):
        world = self.world(route='dense')
        stack = self.stack(world)
        self.do(world, 'prepare', source=world.lease)
        self.do(world, 'prefix', paused=False)

        def explode(layer_idx):
            raise RuntimeError('synthetic enqueue failure')

        stack.dispatcher.lane.layer = explode
        with self.assertRaises(LaneDispatchError):
            world.send('layer', layer=0)
        self.assertIn('layer', stack.dispatcher.quarantined)
        self.assertFalse(stack.dispatcher.drain())
        with self.assertRaises(LaneDispatchError):
            stack.dispatcher.submit(world.commands[-1])
        with self.assertRaises(LaneDispatchError):
            stack.dispatcher.release()
        self.assertEqual(world.accepted[0], [0, 1])

    def test_release_and_reuse_after_queried_drain(self):
        world = self.world(route='dense')
        stack = self.stack(world)
        self.begin_layer0(world)
        command, acks = world.send('layer', layer=0)
        self.expect_pending(world, command, acks)
        stack.ready_attention()
        self.expect_accepted(world, command, world.deliver(command),
                             paused=False)
        self.finish(world)
        self.complete_and_release(world)
        self.assertEqual(stack.factory.released, 1)
        self.assertEqual(stack.dispatcher.accepted, {})
        self.assertIsNone(stack.dispatcher.transaction)
        self.assertIsNone(stack.bridge.transaction)
        self.assertEqual(stack.operation.stats['sparse_source_released'], 0)
        self.assertIsNone(stack.bridge._source_close)

        # The same worker row is reused for the next chunk: a fresh open and a
        # real prepare on the same dispatcher/bridge, no stale op_seq.
        txn2 = et.TransactionId(engine_epoch=1,
                                request_uid=world.txn.request_uid,
                                request_generation=1, chunk_start=0,
                                chunk_end=stack.tokens, txn_seq=2)
        admission = w.AdmissionRecord(
            transaction=txn2, prompt_end=stack.tokens + 1, state_epoch=1,
            committed_end=0, block_tables=((1,),),
            slot_mappings=(tuple(range(stack.tokens)),),
            new_block_ids_to_zero=(1,), cow_pairs=(),
            input_ids=tuple(index % 7 for index in range(stack.tokens)))
        ack = stack.bridge.open(w.OpenRequest(admission, (0,), 1, 8))
        self.assertTrue(ack.accepted)
        lease = w.SourceLeaseDescriptor(txn2, 1, (1,), (0,))
        descriptor = w.SegmentDescriptor(txn2, w.SegmentKind.PREPARE, 0)
        self.assertEqual(w.OpSeqTracker().propose(descriptor), 0)
        prepared = stack.bridge.submit(
            w.SegmentCommand(descriptor, 0, (0,), source=lease))
        self.assertTrue(prepared.accepted)
        self.assertEqual(stack.dispatcher.transaction, txn2)
        self.assertEqual(len(stack.factory.chunks), 2)


# --- sparse selection credit ------------------------------------------------

class SparseSelectionCreditTests(CreditHarness, unittest.TestCase):
    def test_delayed_selection_readback_does_not_consume_accepted_ops(self):
        world = self.world(route='sparse')
        stack = self.stack(world)
        self.begin_layer0(world)
        self.assertEqual(world.accepted[0], [0, 1, 2, 3])
        self.assertEqual(stack.operation.stats['selection_copies'], 1)
        self.assertEqual(stack.operation.stats['buckets'], 0)
        self.assertEqual(stack.operation.stats['sparse_buckets'], 0)
        self.assertEqual(stack.device.host_reads, [])

        command, acks = world.send('layer', layer=0)
        self.expect_pending(world, command, acks)
        credit = stack.driver.submission_credit(0)
        self.assertFalse(credit.can_submit)
        self.assertEqual(credit.reason, 'selection readback not observed')
        self.assertEqual(credit.pending_selection, 1)
        before = self.submission_state(world)
        self.poll(world, command, POLLS)
        self.assertEqual(before, self.submission_state(world))
        # No host read of the pinned scratch before the copy is observed.
        self.assertEqual(stack.device.host_reads, [])
        self.assertEqual(world.accepted[0], [0, 1, 2, 3])

        stack.ready_selection()
        # This submission reads the completed pinned selection and fills the
        # compact bucket; it is real work, so it is accepted (and paused).
        first = world.deliver(command)
        self.expect_accepted(world, command, first, paused=True)
        self.assertEqual(stack.operation.stats['selection_copies'], 1)
        self.assertEqual(stack.operation.stats['sparse_buckets'], 1)
        self.assertEqual(len(stack.device.host_reads), 2)
        self.assertEqual(world.accepted[0], [0, 1, 2, 3, 4])

        # The attention consumer of that bucket is still delayed: the next
        # attempt is typed pending and submits nothing.
        command, acks = world.send('layer', layer=0)
        self.expect_pending(world, command, acks)
        credit = stack.driver.submission_credit(0)
        self.assertEqual(credit.reason, 'attention consumer not observed')
        self.assertEqual(credit.pending_buckets, 1)
        before = self.submission_state(world)
        self.poll(world, command, POLLS)
        self.assertEqual(before, self.submission_state(world))
        self.assertEqual(world.accepted[0], [0, 1, 2, 3, 4])

        stack.ready_attention()
        self.expect_accepted(world, command, world.deliver(command),
                             paused=False)
        self.assertEqual(world.accepted[0], [0, 1, 2, 3, 4, 5])
        self.assertEqual(stack.operation.stats['sparse_buckets'], 1)
        self.assertEqual(stack.attention.post_calls, 1)

        self.finish(world)
        report = self.complete_and_release(world)[0]
        self.assertEqual(report.completed_op_seqs, tuple(world.accepted[0]))
        kinds = [proof.kind for proof in report.proofs]
        self.assertEqual(kinds.count(et.CompletionKind.GPU_EVENT),
                         len(world.accepted[0]))
        self.assertEqual(kinds.count(et.CompletionKind.STATE_WRITE), 1)
        self.assertEqual(stack.operation.stats['sparse_source_released'], 1)
        self.assertTrue(stack.holds[0].released)
        self.assertEqual(stack.holds[0].stats['consumed'], 1)
        self.assertEqual(stack.device.synchronized, 0)

    def test_source_hold_is_not_retired_before_queried_drain(self):
        world = self.world(route='sparse')
        stack = self.stack(world)
        self.begin_layer0(world)
        command, acks = world.send('layer', layer=0)
        self.expect_pending(world, command, acks)
        self.poll(world, command, 50)
        hold = stack.holds[0]
        self.assertFalse(hold.released)
        # The lane still owns the staged selection copy and its source.
        self.assertFalse(stack.dispatcher.drain())
        with self.assertRaises(et.E3Pending):
            stack.dispatcher.release()
        self.assertFalse(hold.released)
        self.assertEqual(stack.operation.stats['sparse_source_released'], 0)


# --- identity / peer / cancellation ----------------------------------------

class IdentityCreditTests(CreditHarness, unittest.TestCase):
    def test_changed_pending_retry_is_refused(self):
        world = self.world(route='dense')
        self.begin_layer0(world)
        command, acks = world.send('layer', layer=0)
        self.expect_pending(world, command, acks)
        other = w.SegmentDescriptor(world.txn, w.SegmentKind.LAYER, 0,
                                    layer_start=1, layer_end=2)
        with self.assertRaises(et.E3ContractError):
            world.trackers[0].propose(other)
        # The unchanged retry still reuses the pending sequence.
        self.assertEqual(world.trackers[0].propose(command.descriptor),
                         command.op_seq)
        self.assertEqual(world.trackers[0].last_accepted, 3)

    def test_identical_replay_is_idempotent_and_changed_reuse_refused(self):
        world = self.world(route='dense')
        stack = self.stack(world)
        self.begin_layer0(world)
        command = world.commands[-1]
        before = self.submission_state(world)
        replay = stack.bridge.submit(command)
        self.assertTrue(replay.accepted)
        self.assertEqual(replay.op_seq, command.op_seq)
        self.assertTrue(replay.paused)
        self.assertEqual(before, self.submission_state(world))

        stored = stack.dispatcher.accepted[command.op_seq][0]
        self.assertEqual(stored.identity(),
                         LocalCommand.from_descriptor(command).identity())
        changed = w.SegmentCommand(
            w.SegmentDescriptor(world.txn, w.SegmentKind.PREFIX, 0),
            command.op_seq, (0,))
        with self.assertRaises(LaneDispatchError) as caught:
            stack.bridge.submit(changed)
        self.assertIn('different command', str(caught.exception))
        self.assertEqual(world.accepted[0], [0, 1, 2, 3])


class PeerCreditTests(CreditHarness, unittest.TestCase):
    def test_one_peer_pending_other_accepted_then_identical_retry(self):
        world = self.world(route='dense', ranks=(0, 1))
        self.begin_layer0(world)
        command, acks = world.send('layer', layer=0)
        self.expect_pending(world, command, acks)

        # One TP peer observes its consumer, the other does not.
        self.stack(world, 1).ready_attention()
        acks = world.deliver(command)
        self.assertTrue(acks[1].accepted)
        self.assertFalse(acks[1].paused)
        self.assertTrue(acks[0].pending)
        self.assertEqual(world.accepted[1], [0, 1, 2, 3, 4])
        self.assertEqual(world.accepted[0], [0, 1, 2, 3])
        self.assertEqual(world.trackers[1].last_accepted, command.op_seq)
        self.assertEqual(world.trackers[0].pending,
                         (command.op_seq, command.descriptor))

        # The accepted peer replays the identical command without executing.
        stack1 = self.stack(world, 1)
        before = self.submission_state(world)
        acks = world.deliver(command)
        self.assertEqual(before, self.submission_state(world))
        self.assertTrue(acks[1].accepted)
        self.assertEqual(acks[1].op_seq, command.op_seq)
        self.assertTrue(acks[0].pending)
        self.assertEqual(stack1.attention.post_calls, 1)
        self.assertEqual(len(stack1.dispatcher.events), 5)

        self.stack(world, 0).ready_attention()
        acks = world.deliver(command)
        self.assertTrue(acks[0].accepted)
        self.assertTrue(acks[1].accepted)
        self.assertEqual(world.accepted,
                         {0: [0, 1, 2, 3, 4], 1: [0, 1, 2, 3, 4]})
        self.assertEqual(world.trackers[0].last_accepted, command.op_seq)
        self.assertEqual(len(self.stack(world, 0).dispatcher.events), 5)


class CancellationCreditTests(CreditHarness, unittest.TestCase):
    def test_cancel_during_credit_wait_submits_nothing_and_retains_owners(self):
        world = self.world(route='dense')
        stack = self.stack(world)
        self.begin_layer0(world)
        command, acks = world.send('layer', layer=0)
        self.expect_pending(world, command, acks)
        events_before = len(stack.dispatcher.events)

        ack = stack.bridge.cancel(w.CancelRequest(world.txn, 1, 'preempt'))
        self.assertFalse(ack.released)
        self.assertFalse(ack.request_closed)
        self.assertEqual(ack.disposition, 'accepted')
        self.assertEqual(ack.completed_op_seqs, ())
        self.assertEqual(ack.pending_op_seqs, (0, 1, 2, 3))
        # No accepted operation or GPU work was added by the cancelled wait.
        self.assertEqual(world.accepted[0], [0, 1, 2, 3])
        self.assertEqual(stack.operation.stats['buckets'], 1)
        self.assertEqual(len(stack.dispatcher.events), events_before)
        with self.assertRaises(LaneDispatchError):
            stack.bridge.submit(command)

        # The bucket's consumer was never observed: ownership is retained.
        self.assertFalse(stack.dispatcher.drain())
        stack.complete_all()
        second = stack.bridge.cancel(w.CancelRequest(world.txn, 1, 'preempt'))
        self.assertTrue(second.released)
        self.assertTrue(second.request_closed)
        self.assertEqual(second.disposition, 'released')
        self.assertEqual(second.completed_op_seqs, (0, 1, 2, 3))
        self.assertEqual(second.pending_op_seqs, ())
        self.assertEqual(second.completed_sequence, 3)


# --- legacy (no seam) source behaviour --------------------------------------

class LegacyCreditTests(CreditHarness, unittest.TestCase):
    def test_legacy_source_reproduces_accepted_metadata_growth(self):
        world = self.world(route='dense', legacy=True)
        stack = self.stack(world)
        self.begin_layer0(world)
        self.assertIsNone(stack.driver.submission_credit(0))
        baseline = (len(stack.device.events),
                    stack.operation.stats['buckets'],
                    len(self.kernel.calls))
        # 71 accepted paused commands for one delayed consumer.
        command, acks = world.send('layer', layer=0)
        self.expect_accepted(world, command, acks, paused=True)
        for _ in range(70):
            # The identical retry of an accepted command is not a new op.
            replay = world.deliver(command)
            self.assertTrue(replay[0].accepted)
            self.assertTrue(replay[0].paused)
            command, acks = world.send('layer', layer=0)
            self.expect_accepted(world, command, acks, paused=True)
        self.assertEqual(len(world.accepted[0]), 4 + 71)
        self.assertEqual(len(set(world.accepted[0])), 4 + 71)
        self.assertEqual(len(stack.dispatcher.events), 4 + 71)
        self.assertEqual(len(stack.dispatcher.lane.segments), 3 + 71)
        # ... and zero new GPU work: this is the source concern on the actual
        # driver path, not a controller receipt.
        self.assertEqual((len(stack.device.events),
                          stack.operation.stats['buckets'],
                          len(self.kernel.calls)), baseline)
        self.assertGreater(len(world.accepted[0]), 64)

    def test_actual_source_field_refuses_65_accepted_sequences(self):
        world = self.world(route='dense', legacy=True)
        stack = self.stack(world)
        self.begin_layer0(world)
        command, acks = world.send('layer', layer=0)
        self.expect_accepted(world, command, acks, paused=True)
        for _ in range(60):
            command, acks = world.send('layer', layer=0)
            self.expect_accepted(world, command, acks, paused=True)
        self.assertEqual(len(world.accepted[0]), 65)
        world.complete_all()
        report = stack.bridge.poll(w.PollRequest(world.txn, 1, 64))
        self.assertEqual(len(report.completed_op_seqs), 65)
        self.assertEqual(len(report.proofs), 65)
        # The engine journal serialiser applies this bound to exactly these
        # two report fields.
        with self.assertRaises(ed.DriverError) as caught:
            ed._bounded_value(list(report.completed_op_seqs))
        self.assertIn('sequence is too long', str(caught.exception))
        with self.assertRaises(ed.DriverError):
            ed._bounded_value([{'kind': proof.kind.value,
                                'op_seq': proof.ticket.op_seq,
                                'completion_seq': proof.completion_seq}
                               for proof in report.proofs])
        # 64 is the actual source-field bound, not a magic larger number.
        self.assertEqual(
            len(ed._bounded_value(list(report.completed_op_seqs[:64]))), 64)

    def test_no_active_driver_keeps_the_plain_lane_behaviour(self):
        world = self.world(route='dense', driverless=True)
        stack = self.stack(world)
        self.expect_accepted(world, *world.send('prepare'))
        self.expect_accepted(world, *world.send('prefix'))
        for layer in (0, 1):
            self.expect_accepted(world, *world.send('layer', layer=layer))
        self.expect_accepted(world, *world.send('suffix'))
        self.expect_accepted(world, *world.send('state_finalize'))
        self.assertEqual(world.accepted[0], [0, 1, 2, 3, 4, 5])
        self.assertEqual(stack.model.forward_layer_calls, [0, 1])
        self.assertIsNone(stack.dispatcher.driver)
        self.assertEqual(stack.operation.stats['buckets'], 0)
        self.assertEqual(stack.attention.pre_calls, 0)

        world.complete_all()
        self.assertTrue(stack.dispatcher.drain())
        self.assertTrue(stack.dispatcher.release())
        self.assertTrue(stack.dispatcher.close_request())


# --- the actual accepted-submission bound -----------------------------------

class AcceptedSubmissionBoundTests(unittest.TestCase):
    """The fixed model plan, not a magic number, bounds the wire metadata."""

    @classmethod
    def setUpClass(cls):
        if not MODEL_CONFIG.is_file():
            raise unittest.SkipTest('reference/model-config.json is unavailable')
        config = json.loads(MODEL_CONFIG.read_text())
        cls.config = config.get('text_config', config)

    def plan(self):
        layer_types = self.config['layer_types']
        layers = len(layer_types)
        positions = self.config['max_position_embeddings']
        pages = -(-positions // PAGE_TOKENS)
        buckets = len(qo.plan_buckets(range(pages)))
        middle = layers // 2
        # The fixed PP split is 24/24 (docs/e3-briefs/
        # claude-20261025-survey_model_layer.md: stage 0 runs layers 0-23,
        # stage 1 runs 24-47).  Derive each stage's QSA/GDN counts from the
        # actual pinned config rather than assuming an even mix.
        # prepare + one PP direction + prefix + suffix + state finalize, then
        # one pre + selection submission + buckets + tail per QSA layer,
        # one segment per GDN. The real delayed-selection test above observes
        # the accepted selection copy separately before the first bucket.
        fixed = 5
        stages = []
        for index, (lo, hi) in enumerate(((0, middle), (middle, layers))):
            types = layer_types[lo:hi]
            qsa = types.count('full_attention')
            gdn = types.count('linear_attention')
            accepted = fixed + gdn + qsa * (3 + buckets)
            stages.append(dict(first=lo, last=hi, qsa=qsa, gdn=gdn,
                               accepted=accepted,
                               # one GPU_EVENT proof per accepted op, the
                               # stage's own STATE_WRITE, and stage 1's
                               # PP_CONSUMER for the received transfer.
                               proofs=accepted + 1 + (1 if index else 0)))
        return dict(layers=layers, positions=positions, pages=pages,
                    buckets=buckets, stages=stages,
                    qsa=layer_types.count('full_attention'),
                    gdn=layer_types.count('linear_attention'))

    def test_fixed_model_maximum_is_59_accepted_operations_per_stage(self):
        plan = self.plan()
        self.assertEqual(plan['layers'], 48)
        self.assertEqual((plan['qsa'], plan['gdn']), (12, 36))
        self.assertEqual(plan['positions'], 262144)
        self.assertEqual((plan['pages'], plan['buckets']), (65, 3))
        stages = plan['stages']
        self.assertEqual([(stage['qsa'], stage['gdn']) for stage in stages],
                         [(6, 18), (6, 18)])
        self.assertEqual([stage['accepted'] for stage in stages], [59, 59])
        self.assertEqual([stage['proofs'] for stage in stages], [60, 61])
        # Both source fields the engine journals stay inside the real bound.
        for stage in stages:
            self.assertEqual(len(ed._bounded_value(
                list(range(stage['accepted'])))), stage['accepted'])
            self.assertEqual(len(ed._bounded_value(
                [{'kind': 'gpu_event', 'op_seq': index,
                  'completion_seq': index}
                 for index in range(stage['proofs'])])), stage['proofs'])
        # A dense (one-bucket) layer is smaller still.
        dense = 5 + stages[0]['gdn'] + stages[0]['qsa'] * 3
        self.assertEqual(dense, 41)
        self.assertLess(dense, max(stage['accepted'] for stage in stages))

    def test_operation_page_cap_is_unreachable_at_the_model_maximum(self):
        plan = self.plan()
        stage = plan['stages'][0]
        self.assertEqual(qo.MAX_LAYER_PAGES, 128)
        # 128 pages needs more positions than the fixed model can ask for, so
        # its 6 buckets / 77 accepted operations are not a legal sparse plan.
        self.assertGreater(qo.MAX_LAYER_PAGES * PAGE_TOKENS,
                           plan['positions'])
        unreachable = 5 + stage['gdn'] + stage['qsa'] * (
            3 + len(qo.plan_buckets(range(qo.MAX_LAYER_PAGES))))
        self.assertGreater(unreachable, 64)
        with self.assertRaises(ed.DriverError):
            ed._bounded_value(list(range(unreachable)))


if __name__ == '__main__':
    unittest.main()
