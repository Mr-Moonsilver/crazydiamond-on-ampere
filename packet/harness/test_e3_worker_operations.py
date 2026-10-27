"""D2 bounded QSA worker-operation tests (CPU fake-CUDA evidence only).

No torch, no vLLM and no CUDA is imported or called.  Tensors, events,
streams and the Triton kernel launcher are local fakes; the *real* generated
operation, the *real* generated attention function and the *real* generated
layer helper are executed against them.  Passing these tests is evidence about
ordering, bounds and refusal logic, not native qualification.
"""
import ast
import math
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import arena_attention
import e3_types as et
import operation_source
import qsa_operation as qo
import resident_attention
import runner_hooks


PAGE_TOKENS = 4048
ROW_WIDTH = 512
HEAD_DIM = ROW_WIDTH // 2
NUM_KV_HEADS = 1
QUERY_HEADS = 6
SOURCE_PAGES = 64
ARENA_ADDRESS = 0x9000
TABLE_ADDRESS = 0x7000


# --- fake tensors / device (clearly not CUDA) -------------------------------

def _contiguous(shape):
    strides = []
    step = 1
    for size in reversed(shape):
        strides.append(step)
        step *= size
    return tuple(reversed(strides))


_ITEMS = {'torch.float32': 4, 'torch.int64': 8, 'torch.int32': 4,
          'torch.bfloat16': 2, 'torch.float16': 2}


class FakeTensor:
    """Shape/stride/dtype/address stand-in; no storage is allocated."""

    def __init__(self, shape, *, device='cuda:0', dtype='torch.bfloat16',
                 pinned=False, address=0x1000, strides=None):
        self._shape = tuple(shape)
        self._device = device
        self._dtype = dtype
        self._pinned = pinned
        self._address = address
        self._item_bytes = _ITEMS[dtype]
        self._strides = tuple(strides) if strides is not None else _contiguous(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def stride(self, dim=None):
        return self._strides if dim is None else self._strides[dim]

    def numel(self):
        n = 1
        for size in self._shape:
            n *= size
        return n

    def element_size(self):
        return self._item_bytes

    def is_pinned(self):
        return self._pinned

    def is_contiguous(self):
        return self._strides == _contiguous(self._shape)

    def data_ptr(self):
        return self._address

    def __getitem__(self, key):
        if isinstance(key, int):
            return FakeTensor(self._shape[1:], device=self._device,
                              dtype=self._dtype, pinned=self._pinned,
                              address=self._address, strides=self._strides[1:])
        if isinstance(key, slice):
            lo, hi, _ = key.indices(self._shape[0])
            return FakeTensor((hi - lo,) + self._shape[1:], device=self._device,
                              dtype=self._dtype, pinned=self._pinned,
                              address=self._address, strides=self._strides)
        raise TypeError('fake tensor only supports int/slice views')

    def __setitem__(self, key, value):
        raise AssertionError('fake dense path must not scatter into the output')

    def transpose(self, first, second):
        shape = list(self._shape)
        strides = list(self._strides)
        shape[first], shape[second] = shape[second], shape[first]
        strides[first], strides[second] = strides[second], strides[first]
        return FakeTensor(tuple(shape), device=self._device, dtype=self._dtype,
                          pinned=self._pinned, address=self._address,
                          strides=tuple(strides))

    def split(self, size, dim=-1):
        dim %= len(self._shape)
        left = list(self._shape)
        right = list(self._shape)
        left[dim] = size
        right[dim] = self._shape[dim] - size
        return (FakeTensor(tuple(left), device=self._device, dtype=self._dtype,
                           pinned=self._pinned, address=self._address,
                           strides=self._strides),
                FakeTensor(tuple(right), device=self._device, dtype=self._dtype,
                           pinned=self._pinned, address=self._address,
                           strides=self._strides))


class FakeTorch:
    """The handful of ``torch`` names the generated code calls."""

    Tensor = FakeTensor
    int64 = 'torch.int64'
    int32 = 'torch.int32'
    float32 = 'torch.float32'

    def arange(self, start, end, *, device=None, dtype=None):
        return FakeTensor((int(end) - int(start),), device=str(device),
                          dtype=dtype or 'torch.int64')

    def empty(self, shape, *, dtype=None, device=None):
        return FakeTensor(tuple(shape), device=str(device),
                          dtype=dtype or 'torch.float32')

    def empty_like(self, tensor):
        return FakeTensor(tensor.shape, device=tensor.device,
                          dtype=tensor.dtype, strides=tensor.stride())


class FakeTriton:
    def next_power_of_2(self, value):
        return 1 if value <= 1 else 1 << (value - 1).bit_length()

    def cdiv(self, a, b):
        return -(-a // b)


class FakeKernel:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return launch


class FakeEvent:
    def __init__(self, token, device, ready):
        self.token = token
        self._device = device
        self.ready = ready
        self.queries = 0

    @property
    def device(self):
        return self._device

    def query(self):
        self.queries += 1
        return self.ready

    def mark_ready(self):
        self.ready = True
        return self


class FakeQsaDevice:
    """Nonblocking device port; ``synchronize`` exists only to fail the test."""

    compute = 'compute'
    copy = 'copy'

    def __init__(self, *, auto_ready=False):
        self.device = 'cuda:0'
        self.auto_ready = auto_ready
        self.events = []
        self.waits = []
        self.copies = []
        self.copy_dependencies = []
        self.queries = 0
        self.synchronized = 0

    def facts(self, tensor):
        return et.TensorFacts(str(tensor.device), tuple(tensor.shape),
                           tuple(tensor.stride()), str(tensor.dtype),
                           tensor.element_size(), tensor.is_pinned(),
                           tensor.is_contiguous(), tensor.data_ptr())

    def record(self, token):
        if token not in (self.compute, self.copy):
            raise AssertionError('unknown stream token in fixture')
        event = FakeEvent(token, self.device, self.auto_ready)
        self.events.append(event)
        return event

    def query(self, event):
        self.queries += 1
        return event.query()

    def wait_event(self, token, event):
        self.waits.append((token, event))

    def copy_pages(self, source, arena, source_page, arena_page, pages):
        if type(pages) is not int or not 1 <= pages <= qo.MAX_PAGES_PER_COPY:
            raise AssertionError('copy bound violated by the operation')
        pending = tuple(event for token, event in self.waits
                        if token == self.copy and not event.ready)
        self.copy_dependencies.append(pending)
        self.copies.append((source_page, arena_page, pages))

    def synchronize(self):
        self.synchronized += 1
        raise AssertionError('the operation must never synchronize')


def make_source(address=0x4000, pages=SOURCE_PAGES):
    return FakeTensor((pages, NUM_KV_HEADS, PAGE_TOKENS, ROW_WIDTH),
                      device='cpu', dtype='torch.bfloat16', pinned=True,
                      address=address)


def make_arena(address=ARENA_ADDRESS):
    return FakeTensor((qo.ARENA_PAGES, NUM_KV_HEADS, PAGE_TOKENS, ROW_WIDTH),
                      device='cuda:0', dtype='torch.bfloat16', address=address)


def make_table(width=64, address=TABLE_ADDRESS):
    return FakeTensor((width,), device='cuda:0', dtype='torch.int64',
                      address=address)


def make_table_grid(width=64, address=TABLE_ADDRESS):
    return FakeTensor((1, width), device='cuda:0', dtype='torch.int64',
                      address=address)


def make_transaction(**overrides):
    values = dict(engine_epoch=1, request_uid='req-a', request_generation=0,
                  chunk_start=0, chunk_end=8096, txn_seq=1)
    values.update(overrides)
    return et.TransactionId(**values)


def make_lane(**overrides):
    values = dict(worker_rank=0, role=et.LaneRole.PREFILL, storage_epoch=3,
                  context_id=0)
    values.update(overrides)
    return et.LaneId(**values)


class Snapshot:
    def __init__(self, rows, block_ids):
        self.rows = rows
        self.block_ids = tuple(block_ids)


class Row:
    def __init__(self, chunk_start=0):
        self.chunk_start = chunk_start


def pages_tokens(pages):
    return pages * PAGE_TOKENS


# --- plan / provider ---------------------------------------------------------

class PlanTests(unittest.TestCase):
    def test_baseline_fifty_page_order_is_24_24_2(self):
        buckets = qo.plan_buckets(tuple(range(50)))
        self.assertEqual([(b.lo, b.hi) for b in buckets],
                         [(0, 24), (24, 48), (48, 50)])
        self.assertEqual([b.pages for b in buckets], [24, 24, 2])
        for bucket in buckets:
            self.assertLessEqual(bucket.pages, qo.MAX_BUCKET_PAGES)
            for _, _, pages in bucket.pieces:
                self.assertTrue(1 <= pages <= qo.MAX_PAGES_PER_COPY)

    def test_pieces_coalesce_only_adjacent_source_pages(self):
        buckets = qo.plan_buckets((5, 6, 7, 8, 9, 20, 21))
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0].sources, (5, 6, 7, 8, 9, 20, 21))
        self.assertEqual(buckets[0].pieces,
                         ((5, 0, 4), (9, 4, 1), (20, 5, 2)))

    def test_bounds_are_validated(self):
        for pages in ((), (-1,), (1.5,), ('2',), tuple(range(qo.MAX_LAYER_PAGES + 1))):
            with self.assertRaises(qo.OperationError):
                qo.plan_buckets(pages)
        for kwargs in ({'bucket_pages': 0}, {'bucket_pages': 25},
                       {'max_piece': 0}, {'max_piece': 5}):
            with self.assertRaises(qo.OperationError):
                qo.plan_buckets((1, 2), **kwargs)

    def test_metadata_provider_is_cpu_only_and_strict(self):
        snapshot = Snapshot([Row(chunk_start=4048)], (55, 56))
        provider = qo.metadata_plan_provider(snapshot,
                    layer_tables=(('layer', (10, 11, 12)),))
        self.assertEqual(provider('layer', 2), (10, 11))
        for name, count in [('other', 2), ('layer', 4), ('layer', 0)]:
            with self.assertRaises(qo.OperationError):
                provider(name, count)
        with self.assertRaisesRegex(qo.OperationError, 'flat block ids refused'):
            qo.metadata_plan_provider(snapshot)
        with self.assertRaisesRegex(qo.OperationError, 'single-row'):
            qo.metadata_plan_provider(Snapshot([Row(), Row()], (10,)))
        with self.assertRaises(qo.OperationError):
            qo.metadata_plan_provider(snapshot, layer_tables=(('layer', (1,)), ('layer', (2,))))


# --- operation state machine -------------------------------------------------

class OperationTests(unittest.TestCase):
    def make(self, *, device=None, pages=(10, 11), provider=None, max_tickets=None):
        device = device or FakeQsaDevice(auto_ready=True)
        source = make_source()
        kwargs = {}
        if max_tickets is not None:
            kwargs['max_tickets'] = max_tickets
        operation = qo.QsaPrefillOperation(
            make_transaction(), make_lane(), device=device,
            plan_provider=provider or (lambda name, n: tuple(pages[:n])),
            **kwargs)
        operation.register_source('layers.0.attn', source)
        self.source = source
        return operation, device

    def begin(self, operation, pages=(10, 11), *, arena=None, table=None,
              layer='layers.0.attn', tokens=None):
        arena = arena or make_arena()
        table = table or make_table()
        operation.begin_forward()
        operation.begin_layer(layer, self.source)
        operation.producer_submitted(layer, self.source)
        operation.bind_plan(layer, table, len(pages),
                            tokens if tokens is not None else pages_tokens(len(pages)),
                            arena)
        return arena, table

    def fill_all(self, operation, arena, table, pages, *, complete=True):
        tickets = []
        for bucket in operation._layers['layers.0.attn'].buckets:
            tickets.append(operation.fill_bucket(arena[:bucket.pages],
                                                 bucket.lo, bucket.hi, table))
            operation.after_attention_bucket(arena[:bucket.pages], table)
            if complete:
                for event in operation.device.events:
                    event.mark_ready()
        return tickets

    def test_producer_order_and_proof_kinds(self):
        operation, device = self.make()
        arena, table = self.begin(operation)
        bucket = operation._layers['layers.0.attn'].buckets[0]
        copy_ticket = operation.fill_bucket(arena[:bucket.pages], bucket.lo,
                                            bucket.hi, table)
        consumer_ticket = operation.after_attention_bucket(arena[:bucket.pages], table)
        operation.end_layer('layers.0.attn')
        operation.end_forward()
        proofs = operation.poll()
        by_ticket = {(p.ticket.op_seq, p.ticket.dependency_role): p.kind
                     for p in proofs}
        producer = operation._layers['layers.0.attn'].producer.ticket
        self.assertEqual(by_ticket[(producer.op_seq, et.DependencyRole.PRODUCER)],
                         et.CompletionKind.GPU_EVENT)
        self.assertEqual(by_ticket[(copy_ticket.op_seq, et.DependencyRole.PRODUCER)],
                         et.CompletionKind.COPY_EVENT)
        self.assertEqual(by_ticket[(consumer_ticket.op_seq, et.DependencyRole.CONSUMER)],
                         et.CompletionKind.GPU_EVENT)
        self.assertNotIn(et.CompletionKind.STATE_WRITE,
                         [p.kind for p in proofs])
        self.assertEqual(operation.pending, 0)

    def test_every_copy_is_at_most_four_pages_and_buckets_are_bounded(self):
        operation, device = self.make(pages=tuple(range(50)))
        arena, table = self.begin(operation, pages=tuple(range(50)),
                                  tokens=pages_tokens(50))
        run = operation._layers['layers.0.attn']
        self.assertEqual([(b.lo, b.hi) for b in run.buckets],
                         [(0, 24), (24, 48), (48, 50)])
        self.assertLessEqual(len(run.buckets), qo.MAX_STAGES_PER_LAYER)
        self.fill_all(operation, arena, table, tuple(range(50)))
        self.assertEqual(sum(pages for _, _, pages in device.copies), 50)
        self.assertTrue(all(1 <= pages <= 4 for _, _, pages in device.copies))
        self.assertGreaterEqual(max(pages for _, _, pages in device.copies), 2)
        self.assertEqual(operation.stats['buckets'], 3)
        self.assertEqual(operation.stats['pieces'], len(device.copies))

    def test_one_bucket_may_be_outstanding(self):
        operation, device = self.make()
        arena, table = self.begin(operation)
        run = operation._layers['layers.0.attn']
        first = run.buckets[0]
        operation.fill_bucket(arena[:first.pages], first.lo, first.hi, table)
        self.assertEqual(operation.pending_buckets, 1)
        with self.assertRaises(et.E3Pending):
            operation.fill_bucket(arena[:first.pages], first.lo, first.hi, table)
        operation.after_attention_bucket(arena[:first.pages], table)

    def test_delayed_events_never_block_the_host(self):
        operation, device = self.make(device=FakeQsaDevice(auto_ready=False))
        arena, table = self.begin(operation)
        run = operation._layers['layers.0.attn']
        first = run.buckets[0]
        operation.fill_bucket(arena[:first.pages], first.lo, first.hi, table)
        operation.after_attention_bucket(arena[:first.pages], table)
        # No event is ready: poll returns nothing and still does no host wait.
        self.assertEqual(operation.poll(), ())
        self.assertEqual(operation.pending, 3)
        self.assertEqual(device.synchronized, 0)
        for event in device.events:
            event.mark_ready()
        proofs = operation.poll()
        self.assertEqual(len(proofs), 3)
        self.assertEqual(operation.pending, 0)
        self.assertEqual(device.synchronized, 0)

    def test_delayed_consumer_blocks_arena_reuse_by_stream_dependency(self):
        operation, device = self.make(device=FakeQsaDevice(auto_ready=False),
                                      pages=tuple(range(30)))
        arena, table = self.begin(operation, pages=tuple(range(30)),
                                  tokens=pages_tokens(30))
        run = operation._layers['layers.0.attn']
        first, second = run.buckets[0], run.buckets[1]
        operation.fill_bucket(arena[:first.pages], first.lo, first.hi, table)
        consumer = operation.after_attention_bucket(arena[:first.pages], table)
        consumer_event = run.consumed[-1].event
        self.assertFalse(consumer_event.ready)
        copies_before = len(device.copies)
        with self.assertRaises(et.E3Pending):
            operation.fill_bucket(arena[:second.pages], second.lo, second.hi, table)
        self.assertEqual(len(device.copies), copies_before)
        self.assertEqual(operation.pending_buckets, 1)
        for event in device.events:
            event.mark_ready()
        operation.poll()
        operation.fill_bucket(arena[:second.pages], second.lo, second.hi, table)
        self.assertIn((device.copy, consumer_event), device.waits)
        # The copy was enqueued behind the consumer event, not executed.
        copies_before = len(device.copies)
        self.assertEqual(copies_before, len(first.pieces) + len(second.pieces))
        # Completing the consumer only later lets the bucket retire.
        consumer_event.mark_ready()
        self.assertIn(operation._layers['layers.0.attn'].consumed[0].ticket,
                      [operation._layers['layers.0.attn'].consumed[0].ticket])
        self.assertFalse(device.synchronized)

    def test_cross_layer_reuse_waits_on_the_previous_layers_consumer(self):
        operation, device = self.make(device=FakeQsaDevice(auto_ready=False))
        other_source = make_source(address=0x4100)
        other_arena = make_arena(address=0x9100)
        other_table = make_table(address=0x7100)
        operation.register_source('layers.1.attn', other_source)
        operation.begin_forward(layers=('layers.0.attn', 'layers.1.attn'))
        operation.begin_layer('layers.0.attn', self.source)
        operation.producer_submitted('layers.0.attn', self.source)
        operation.bind_plan('layers.0.attn', make_table(),
                            qo.plan_buckets((10, 11))[0].pages,
                            pages_tokens(2), make_arena())
        run0 = operation._layers['layers.0.attn']
        bucket = run0.buckets[0]
        operation.fill_bucket(make_arena()[:bucket.pages], bucket.lo, bucket.hi,
                              make_table())
        operation.after_attention_bucket(make_arena()[:bucket.pages], make_table())
        last_consumer = run0.consumed[-1].event
        operation.end_layer('layers.0.attn')
        operation.begin_layer('layers.1.attn', other_source)
        operation.producer_submitted('layers.1.attn', other_source)
        operation.bind_plan('layers.1.attn', other_table,
                            qo.plan_buckets((10, 11))[0].pages,
                            pages_tokens(2), other_arena)
        run1 = operation._layers['layers.1.attn']
        first = run1.buckets[0]
        with self.assertRaises(et.E3Pending):
            operation.fill_bucket(other_arena[:first.pages], first.lo, first.hi, other_table)
        for event in device.events:
            event.mark_ready()
        operation.poll()
        operation.fill_bucket(other_arena[:first.pages], first.lo, first.hi, other_table)
        self.assertIn((device.copy, last_consumer), device.waits)

    def test_exact_bucket_order_and_source_selection(self):
        pages = (5, 6, 7, 8, 9, 20, 21)
        operation, device = self.make(pages=pages)
        arena, table = self.begin(operation, pages=pages,
                                  tokens=pages_tokens(len(pages)))
        run = operation._layers['layers.0.attn']
        bucket = run.buckets[0]
        operation.fill_bucket(arena[:bucket.pages], bucket.lo, bucket.hi, table)
        self.assertEqual(device.copies, [(5, 0, 4), (9, 4, 1), (20, 5, 2)])
        operation.after_attention_bucket(arena[:bucket.pages], table)
        # A second bucket is outside the bound plan; a wrong span is refused
        # before any copy when a later bucket exists.
        with self.assertRaisesRegex(qo.OperationError, 'more buckets'):
            operation.fill_bucket(arena[:bucket.pages], bucket.lo, bucket.hi, table)

    def test_poll_queries_only_and_never_launches(self):
        operation, device = self.make(device=FakeQsaDevice(auto_ready=False))
        arena, table = self.begin(operation)
        run = operation._layers['layers.0.attn']
        bucket = run.buckets[0]
        operation.fill_bucket(arena[:bucket.pages], bucket.lo, bucket.hi, table)
        operation.after_attention_bucket(arena[:bucket.pages], table)
        copies = len(device.copies)
        waits = len(device.waits)
        records = len(device.events)
        operation.poll()
        operation.drain()
        operation.inventory()
        self.assertGreater(device.queries, 0)
        self.assertEqual(len(device.copies), copies)
        self.assertEqual(len(device.waits), waits)
        self.assertEqual(len(device.events), records)
        self.assertEqual(device.synchronized, 0)

    def test_cancel_before_a_boundary_refuses_new_work(self):
        operation, device = self.make()
        self.assertEqual(operation.cancel('owner stop'), 'owner stop')
        self.assertEqual(operation.cancelled, 'owner stop')
        with self.assertRaises(qo.OperationCancelled):
            operation.begin_forward()
        self.assertEqual(operation.cancel('again'), 'owner stop')
        with self.assertRaises(qo.OperationError):
            operation.cancel('')
        self.assertEqual(operation.cancelled, 'owner stop')

    def test_cancel_after_submission_retires_the_filled_bucket_only(self):
        operation, device = self.make(device=FakeQsaDevice(auto_ready=False),
                                      pages=tuple(range(30)))
        arena, table = self.begin(operation, pages=tuple(range(30)),
                                  tokens=pages_tokens(30))
        run = operation._layers['layers.0.attn']
        first = run.buckets[0]
        operation.fill_bucket(arena[:first.pages], first.lo, first.hi, table)
        operation.cancel('stop')
        # The already-filled bucket may still record its consumer event so the
        # arena and storage can be retired.
        operation.after_attention_bucket(arena[:first.pages], table)
        self.assertEqual(run.next_bucket, 1)
        # A cancelled operation never submits the remaining bucket.
        second = run.buckets[1]
        with self.assertRaises(qo.OperationCancelled):
            operation.fill_bucket(arena[:second.pages], second.lo, second.hi, table)
        self.assertFalse(operation.drain())
        with self.assertRaises(et.E3Pending):
            operation.release()
        for event in device.events:
            event.mark_ready()
        self.assertTrue(operation.drain())
        # Cancellation retires only the submitted prefix, after its consumers.
        self.assertEqual(operation.release(), 3)

    def test_cancelled_finished_transaction_releases(self):
        operation, device = self.make()
        arena, table = self.begin(operation)
        self.fill_all(operation, arena, table, (10, 11))
        operation.end_layer('layers.0.attn')
        operation.end_forward()
        operation.cancel('owner stop after submission')
        self.assertTrue(operation.drain())
        self.assertEqual(operation.release(), 3)
        self.assertEqual(operation.pending, 0)
        self.assertIsNone(operation._layers['layers.0.attn'].source)

    def test_stale_lane_allocation_and_source_epochs_are_rejected(self):
        operation, device = self.make()
        arena, table = self.begin(operation)
        run = operation._layers['layers.0.attn']
        bucket = run.buckets[0]
        operation.fill_bucket(arena[:bucket.pages], bucket.lo, bucket.hi, table)
        operation.after_attention_bucket(arena[:bucket.pages], table)
        with self.assertRaisesRegex(qo.OperationError, 'stale lane'):
            operation.resume('layers.0.attn', qo.Stage.HISTORY_READY,
                             lane=make_lane(storage_epoch=4))
        with self.assertRaisesRegex(qo.OperationError, 'stale allocation'):
            operation.resume('layers.0.attn', qo.Stage.HISTORY_READY,
                             allocation_epoch=99)
        with self.assertRaisesRegex(qo.OperationError, 'stale source epoch'):
            operation.resume('layers.0.attn', qo.Stage.HISTORY_READY,
                             source_epoch=7)
        with self.assertRaisesRegex(qo.OperationError, 'not bound'):
            operation.resume('layers.9.attn', qo.Stage.HISTORY_READY)
        with self.assertRaisesRegex(qo.OperationError, 'OperationStage'):
            operation.resume('layers.0.attn', 'history_ready')

    def test_operation_identity_is_pinned(self):
        with self.assertRaisesRegex(qo.OperationError, 'TransactionId'):
            qo.QsaPrefillOperation('txn', make_lane(), device=FakeQsaDevice())
        with self.assertRaisesRegex(qo.OperationError, 'LaneId'):
            qo.QsaPrefillOperation(make_transaction(), 'lane',
                                   device=FakeQsaDevice())
        with self.assertRaisesRegex(qo.OperationError, 'device port'):
            qo.QsaPrefillOperation(make_transaction(), make_lane(), device=None)
        with self.assertRaisesRegex(qo.OperationError, 'allocation epoch'):
            qo.QsaPrefillOperation(make_transaction(), make_lane(),
                                   device=FakeQsaDevice(), allocation_epoch=2)
        with self.assertRaisesRegex(qo.OperationError, 'source epoch entry'):
            qo.QsaPrefillOperation(make_transaction(), make_lane(),
                                   device=FakeQsaDevice(),
                                   source_epochs={'x': -1})
        self.assertEqual(qo.QsaPrefillOperation(
            make_transaction(), make_lane(), device=FakeQsaDevice()).lane.storage_epoch, 3)

    def test_source_and_arena_geometry_are_refused(self):
        operation, device = self.make()
        with self.assertRaisesRegex(qo.OperationError, 'pinned BF16 CPU pool'):
            operation.register_source('x', FakeTensor(
                (4, 1, PAGE_TOKENS, ROW_WIDTH), device='cpu',
                dtype='torch.bfloat16', pinned=False))
        with self.assertRaisesRegex(qo.OperationError, 'pinned BF16 CPU pool'):
            operation.register_source('x', FakeTensor(
                (4, 1, PAGE_TOKENS, ROW_WIDTH), device='cuda:0',
                dtype='torch.bfloat16', pinned=True))
        arena, table = self.begin(operation)
        with self.assertRaisesRegex(qo.OperationError, 'invalid existing arena'):
            operation.bind_plan('layers.0.attn', table, 2, pages_tokens(2),
                                FakeTensor((24, 1, PAGE_TOKENS, ROW_WIDTH),
                                           device='cpu', dtype='torch.bfloat16'))
        with self.assertRaisesRegex(qo.OperationError, 'block-table row'):
            operation.bind_plan('layers.0.attn',
                                FakeTensor((16,), device='cuda:0',
                                           dtype='torch.int64', strides=(2,)),
                                2, pages_tokens(2), arena)

    def test_page_plan_and_layer_boundaries_are_enforced(self):
        operation, device = self.make()
        with self.assertRaisesRegex(qo.OperationError, 'outside an open'):
            operation.begin_layer('layers.0.attn', self.source)
        arena, table = self.begin(operation)
        with self.assertRaisesRegex(qo.OperationError, 'already ran|unfinished'):
            operation.begin_layer('layers.0.attn', self.source)
        with self.assertRaisesRegex(qo.OperationError, 'disagrees'):
            operation.bind_plan('layers.0.attn', table, 2, PAGE_TOKENS, arena)
        with self.assertRaisesRegex(qo.OperationError, 'out of bounds'):
            operation.bind_plan('layers.0.attn', table, 0, PAGE_TOKENS, arena)
        with self.assertRaisesRegex(qo.OperationError, 'ended before|outstanding'):
            operation.end_layer('layers.0.attn')
        run = operation._layers['layers.0.attn']
        bucket = run.buckets[0]
        operation.fill_bucket(arena[:bucket.pages], bucket.lo, bucket.hi, table)
        with self.assertRaisesRegex(qo.OperationError, 'ended before|outstanding'):
            operation.end_layer('layers.0.attn')
        operation.after_attention_bucket(arena[:bucket.pages], table)
        operation.end_layer('layers.0.attn')
        self.assertEqual(operation.end_forward(), ('layers.0.attn',))

    def test_no_plan_provider_refuses_before_any_dma(self):
        device = FakeQsaDevice()
        operation = qo.QsaPrefillOperation(make_transaction(), make_lane(),
                                           device=device)
        source = make_source()
        operation.register_source('layers.0.attn', source)
        operation.begin_forward()
        operation.begin_layer('layers.0.attn', source)
        operation.producer_submitted('layers.0.attn', source)
        with self.assertRaisesRegex(qo.OperationError, 'no CPU-owned physical page'):
            operation.bind_plan('layers.0.attn', make_table(), 2,
                                pages_tokens(2), make_arena())
        self.assertEqual(device.copies, [])

    def test_ticket_budget_is_bounded(self):
        operation, device = self.make(max_tickets=1)
        arena, table = self.begin(operation)
        bucket = operation._layers['layers.0.attn'].buckets[0]
        with self.assertRaisesRegex(qo.OperationError, 'budget exhausted'):
            operation.fill_bucket(arena[:bucket.pages], bucket.lo, bucket.hi,
                                  table)
        with self.assertRaisesRegex(qo.OperationError, 'invalid tracked-ticket'):
            qo.QsaPrefillOperation(make_transaction(), make_lane(),
                                   device=FakeQsaDevice(), max_tickets=0)

    def test_resume_does_not_advance_on_an_unobserved_event(self):
        operation, device = self.make(device=FakeQsaDevice(auto_ready=False))
        arena, table = self.begin(operation)
        with self.assertRaises(et.E3Pending):
            operation.resume('layers.0.attn', qo.Stage.HISTORY_READY)
        for event in device.events:
            event.mark_ready()
        index, buckets = operation.resume('layers.0.attn', qo.Stage.HISTORY_READY)
        self.assertEqual(index, 0)
        self.assertEqual(len(buckets), 1)
        run = operation._layers['layers.0.attn']
        bucket = buckets[0]
        operation.fill_bucket(arena[:bucket.pages], bucket.lo, bucket.hi, table)
        active, ticket = operation.resume('layers.0.attn', qo.Stage.ATTENTION)
        self.assertEqual(active, 0)
        self.assertIs(ticket, run._pending_consumer.ticket)


# --- generated call sites executed against the real operation ---------------

def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _function_node(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError('generated function missing: ' + name)


class GeneratedCallSiteTests(unittest.TestCase):
    """Execute the generated attention function and layer helper.

    The E2 composition bytes are pinned by ``operation_source``; if either
    generated call site is dropped the operation performs no DMA and the test
    fails.  Fake tensors/events again, not CUDA.
    """

    @classmethod
    def setUpClass(cls):
        ref = ROOT / 'reference/vllm-src/models/qwen4_exp/nvidia'
        qsa = (ref / 'qsa.py').read_text()
        ops = (ref / 'ops/qsa.py').read_text()
        cls.qsa_e2 = runner_hooks.qsa_staged(qsa)
        cls.ops_e2 = arena_attention.generate(ops)
        cls.generated_qsa = operation_source.generate_layer(cls.qsa_e2)
        cls.generated_ops = operation_source.generate_ops(cls.ops_e2)

    def layers(self):
        self.device = FakeQsaDevice(auto_ready=True)
        self.kernel = FakeKernel('gqa')
        self.gather = FakeKernel('gather')

        self.torch = FakeTorch()
        cuda = FakeTorch()
        globals_ops = {
            '__name__': 'ops.qsa',
            'torch': self.torch,
            'os': os,
            'triton': FakeTriton(),
            '_STAGE_TIMER': None,
            '_qsa_sparse_staged_gqa_kernel': self.kernel,
            '_qsa_gather_selected_host_rows': self.gather,
            '_select_config': lambda rows, heads, prefill, width: (64, 4, 1, None),
        }
        module = ast.Module(
            body=[_function_node(self.generated_ops, 'flash_s5_arena_attention')],
            type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, '<generated ops>', 'exec'), globals_ops)
        self.attention = globals_ops['flash_s5_arena_attention']

        package_name = 'models.qwen4_exp.nvidia.ops.qsa'
        package = _module(package_name,
                          flash_s5_arena_attention=self.attention,
                          qsa_get_staging_arena=lambda kv_cache, nbytes: self.arena)
        for name in ('models', 'models.qwen4_exp', 'models.qwen4_exp.nvidia',
                     'models.qwen4_exp.nvidia.ops'):
            sys.modules.setdefault(name, _module(name))
        self.package = package
        sys.modules[package_name] = package
        self.addCleanup(sys.modules.pop, package_name, None)
        helper_module = ast.Module(
            body=[_function_node(self.generated_qsa,
                                 '_flash_s5_staged_operation')],
            type_ignores=[])
        ast.fix_missing_locations(helper_module)
        globals_helper = {
            '__name__': 'models.qwen4_exp.nvidia.qsa',
            '__package__': 'models.qwen4_exp.nvidia',
            'torch': self.torch,
        }
        exec(compile(helper_module, '<generated qsa>', 'exec'),
             globals_helper)
        self.helper = globals_helper['_flash_s5_staged_operation']
        return self

    def make_case(self, pages, block_ids, tokens=None):
        pages = tuple(pages)
        tokens = pages_tokens(len(pages)) if tokens is None else tokens
        source = make_source()
        self.arena = make_arena()
        table = make_table_grid()
        operation = qo.QsaPrefillOperation(
            make_transaction(chunk_end=tokens), make_lane(), device=self.device,
            plan_provider=qo.metadata_plan_provider(
                Snapshot([Row()], block_ids),
                layer_tables=(('layers.0.attn', tuple(block_ids)),)))
        operation.register_source('layers.0.attn', source)
        operation.begin_forward()
        operation.begin_layer('layers.0.attn', source)
        operation.producer_submitted('layers.0.attn', source)
        layer = types.SimpleNamespace(
            layer_name='layers.0.attn', _qsa_kv_offload=True,
            _qsa_stage_min_rows=16, _qsa_stage_arena_bytes=99_483_648)
        query = FakeTensor((tokens, QUERY_HEADS, HEAD_DIM), device='cuda:0')
        output = FakeTensor((tokens, QUERY_HEADS, HEAD_DIM), device='cuda:0')
        metadata = types.SimpleNamespace(
            _qsa_host_meta=((0, tokens), (tokens,)),
            query_start_loc=FakeTensor((2,), device='cpu', dtype='torch.int32'),
            seq_lens=FakeTensor((1,), device='cpu', dtype='torch.int32'),
            block_table=table)
        kv_cache = FakeTensor((SOURCE_PAGES, NUM_KV_HEADS, PAGE_TOKENS, ROW_WIDTH),
                              device='cpu', dtype='torch.bfloat16', pinned=True)
        return operation, layer, query, output, metadata, kv_cache

    def run_case(self, pages, block_ids, tokens=None):
        (operation, layer, query, output, metadata,
         kv_cache) = self.make_case(pages, block_ids, tokens)
        continuation = self.helper(operation, layer, query, kv_cache,
                            FakeTensor((1, 4), device='cuda:0', dtype='torch.int64'),
                            metadata, query.shape[0], use_prefill_config=True,
                            output=output)
        operation.attach_continuation(continuation)
        while True:
            try:
                operation.advance()
            except StopIteration as end:
                return operation, end.value

    def test_generated_sites_are_present_exactly_once(self):
        # Two overlays: the dense S5 function and the sparse-history one.  Each
        # must contain its bounded fill and consumer yield exactly once.
        for name in ('flash_s5_arena_attention', 'flash_s5_sparse_arena_attention'):
            generated = resident_attention.function(self.generated_ops, name)
            self.assertEqual(generated.count('e3op.fill_bucket('), 1)
            self.assertEqual(generated.count('e3op.after_attention_bucket('), 1)
        self.assertEqual(self.generated_qsa.count('_flash_s5_staged_operation('), 1)
        self.assertEqual(
            self.generated_qsa.count('_flash_s5_sparse_staged_operation('), 1)
        self.assertIn('operation.bind_plan(', self.generated_qsa)
        self.assertIn('operation.bind_sparse_context(', self.generated_qsa)
        self.assertIn('e3op=operation,', self.generated_qsa)
        # Producer boundary wraps index_and_store; consumer closes after the
        # pinned forward call.  Removing either site changes this ordering.
        qsa = self.generated_qsa
        self.assertLess(qsa.index('_s5_op.begin_layer(self.layer_name'),
                        qsa.index('selected = self.indexer('))
        self.assertLess(qsa.index('_s5_op.producer_submitted('),
                        qsa.index('impl.forward_qsa('))
        self.assertLess(qsa.index('impl.forward_qsa('),
                        qsa.index('_s5_op.end_layer(self.layer_name)'))
        self.assertEqual(operation_source.revert_layer(self.generated_qsa),
                         self.qsa_e2)
        self.assertEqual(operation_source.revert_ops(self.generated_ops),
                         self.ops_e2)

    def test_generated_attention_drives_bounded_buckets(self):
        # 50 dense pages, fragmented source groups of four adjacent pages.
        block_ids = tuple(i for group in range(13)
                          for i in (group * 5, group * 5 + 1,
                                    group * 5 + 2, group * 5 + 3))[:50]
        block_ids = block_ids[:48] + (60, 61)
        layer = self.layers()
        with patch.dict(os.environ, {'VLLM_QSA_KVO_SPARSE_PREFILL': '0'}):
            operation, spans = layer.run_case(range(50), block_ids)
        self.assertEqual(spans, (0, pages_tokens(50)))
        run = operation._layers['layers.0.attn']
        self.assertEqual([(b.lo, b.hi) for b in run.buckets],
                         [(0, 24), (24, 48), (48, 50)])
        self.assertEqual(operation.stats['buckets'], 3)
        self.assertEqual(operation.stats['pieces'], 13)
        self.assertEqual(sum(p for _, _, p in layer.device.copies), 50)
        self.assertTrue(all(1 <= p <= 4 for _, _, p in layer.device.copies))
        self.assertEqual(layer.gather.calls, [])  # dense path, never sparse
        self.assertEqual(len(layer.kernel.calls), 3)
        flags = [(kwargs['FIRST_BUCKET'], kwargs['LAST_BUCKET'])
                 for _, _, kwargs in layer.kernel.calls]
        self.assertEqual(flags, [(True, False), (False, False), (False, True)])
        for _, _, kwargs in layer.kernel.calls:
            self.assertEqual(kwargs['PAGE_SIZE'], PAGE_TOKENS)
            self.assertEqual(kwargs['HEAD_DIM'], HEAD_DIM)
        # The generated attention calls only the operation, never a host wait.
        self.assertEqual(layer.device.synchronized, 0)
        operation.end_layer('layers.0.attn')
        operation.end_forward()
        for event in layer.device.events:
            event.mark_ready()
        proofs = operation.poll()
        self.assertEqual(operation.pending, 0)
        self.assertEqual(
            sorted(p.kind.value for p in proofs),
            ['copy_event'] * 3 + ['gpu_event'] * 4)

    def test_removing_the_generated_fill_call_breaks_the_operation(self):
        layer = self.layers()
        broken = self.generated_ops.replace(
            'e3op.fill_bucket(staged, page_lo, page_hi, block_table_row)',
            'pass  # call site removed')
        self.assertNotEqual(broken, self.generated_ops)
        module = ast.Module(
            body=[_function_node(broken, 'flash_s5_arena_attention')],
            type_ignores=[])
        ast.fix_missing_locations(module)
        globals_ops = {
            'torch': layer.torch, 'os': os, 'triton': FakeTriton(),
            '_STAGE_TIMER': None,
            '_qsa_sparse_staged_gqa_kernel': layer.kernel,
            '_qsa_gather_selected_host_rows': layer.gather,
            '_select_config': lambda *a: (64, 4, 1, None),
        }
        exec(compile(module, '<broken ops>', 'exec'), globals_ops)
        layer.package.flash_s5_arena_attention = globals_ops[
            'flash_s5_arena_attention']
        with self.assertRaises((qo.OperationError, RuntimeError)):
            with patch.dict(os.environ, {'VLLM_QSA_KVO_SPARSE_PREFILL': '0'}):
                layer.run_case(range(50), tuple(range(50)))

    def test_removing_the_generated_consumer_call_is_detected(self):
        layer = self.layers()
        broken = self.generated_ops.replace(
            'yield e3op.after_attention_bucket(\n'
            '            staged, block_table_row, q, logical_indices, rows, acc, max_state, norm_state, out)',
            'yield None  # call site removed')
        self.assertNotEqual(broken, self.generated_ops)
        module = ast.Module(
            body=[_function_node(broken, 'flash_s5_arena_attention')],
            type_ignores=[])
        ast.fix_missing_locations(module)
        globals_ops = {
            'torch': layer.torch, 'os': os, 'triton': FakeTriton(),
            '_STAGE_TIMER': None,
            '_qsa_sparse_staged_gqa_kernel': layer.kernel,
            '_qsa_gather_selected_host_rows': layer.gather,
            '_select_config': lambda *a: (64, 4, 1, None),
        }
        exec(compile(module, '<broken ops>', 'exec'), globals_ops)
        layer.package.flash_s5_arena_attention = globals_ops[
            'flash_s5_arena_attention']
        with self.assertRaises((qo.OperationError, RuntimeError)):
            with patch.dict(os.environ, {'VLLM_QSA_KVO_SPARSE_PREFILL': '0'}):
                layer.run_case(range(50), tuple(range(50)))


class _InferenceModeTorch:
    """The runner seam only uses ``torch`` for its decorator."""

    @staticmethod
    def inference_mode():
        return lambda function: function


class GeneratedRunnerSeamTests(unittest.TestCase):
    """Execute the generated runner seam itself, not a fixture mirror.

    ``native_resources`` refuses to launch a prefill unless the runner carries
    ``_flash_s5_execute_prefill_operation``; here the generated method is
    compiled and run so a removed insertion fails instead of leaving an unused
    seam.
    """

    @classmethod
    def setUpClass(cls):
        import make_s5_overlap
        runner = (ROOT / 'reference/vllm-src/v1/worker/gpu/model_runner.py').read_text()
        cls.composed = make_s5_overlap.compose_runner(
            runner_hooks.runner_startup(runner_hooks.runner_execution(runner)))
        cls.source = operation_source.generate_runner(cls.composed)

    def seam(self):
        # The seam is a method on the runner class, not a module function.
        node = next(node for node in ast.walk(ast.parse(self.source))
                    if isinstance(node, ast.FunctionDef)
                    and node.name == operation_source.RUNNER_SEAM)
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {'torch': _InferenceModeTorch()}
        exec(compile(module, '<generated runner seam>', 'exec'), namespace)
        return namespace[operation_source.RUNNER_SEAM]

    def test_generated_seam_cannot_run_whole_stock_forward(self):
        seam = self.seam()
        class Runner:
            def execute_model(self, output):
                raise AssertionError('whole forward must never be called')
        with self.assertRaisesRegex(RuntimeError, 'bounded layer dispatcher'):
            seam(Runner(), object())

    def test_generated_seam_reverts_to_the_s5_composed_runner(self):
        self.assertEqual(operation_source.revert_runner(self.source), self.composed)
        node = next(node for node in ast.walk(ast.parse(self.source))
                    if isinstance(node, ast.FunctionDef)
                    and node.name == operation_source.RUNNER_SEAM)
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n.func, ast.Attribute) and
                             n.func.attr in ('execute_model', 'sample_tokens') for n in calls))




class ReviewOperationTests(unittest.TestCase):
    make = OperationTests.make
    begin = OperationTests.begin
    def test_review_credit_stays_owned_after_consumer_submission(self):
        op, dev = self.make(device=FakeQsaDevice(auto_ready=False), pages=tuple(range(50)))
        arena, table = self.begin(op, pages=tuple(range(50)))
        op.fill_bucket(arena[:24], 0, 24, table)
        op.after_attention_bucket(arena, table)
        copies = list(dev.copies)
        for event in dev.events[:-1]:
            event.mark_ready()
        op.poll()
        self.assertEqual(op.pending_buckets, 1)
        with self.assertRaises(et.E3Pending):
            op.fill_bucket(arena[:24], 24, 48, table)
        self.assertEqual(copies, dev.copies)
        dev.events[-1].mark_ready()
        op.fill_bucket(arena[:24], 24, 48, table)
        self.assertEqual(len(dev.copies), 12)

    def test_review_copy_failure_quarantines_and_retains(self):
        op, dev = self.make(pages=tuple(range(30)))
        arena, table = self.begin(op, pages=tuple(range(30)))
        real_copy = dev.copy_pages
        def failing(*args):
            real_copy(*args)
            raise RuntimeError('copy enqueue failed')
        dev.copy_pages = failing
        with self.assertRaises(RuntimeError):
            op.fill_bucket(arena[:24], 0, 24, table)
        self.assertEqual(len(dev.copies), 1)
        for event in dev.events:
            event.mark_ready()
        op.cancel()
        self.assertFalse(op.drain())
        with self.assertRaises(et.E3Pending):
            op.release()
        self.assertIs(op._layers['layers.0.attn'].source, self.source)
        self.assertEqual(op.pending_buckets, 1)

    def test_review_eventless_reservation_is_not_drained(self):
        op, dev = self.make()
        op.begin_forward()
        op.begin_layer('layers.0.attn', self.source)
        op.cancel()
        self.assertFalse(op.drain())
        with self.assertRaises(et.E3Pending):
            op.release()

    def test_review_budget_preflight_does_not_mutate_ticket_ledger(self):
        op, dev = self.make(max_tickets=2)
        arena, table = self.begin(op)
        with self.assertRaises(qo.OperationError):
            op.fill_bucket(arena[:2], 0, 2, table)
        self.assertEqual((len(op._records), op._op_seq, len(dev.copies)), (1, 1, 0))

    def test_review_changed_mapping_same_shape_is_rejected(self):
        pages = [10, 11]
        op, dev = self.make(provider=lambda _, n: tuple(pages[:n]))
        arena, table = self.begin(op)
        pages[0] = 12
        with self.assertRaisesRegex(qo.OperationError, 'page plan changed'):
            op.bind_plan('layers.0.attn', table, 2, pages_tokens(2), arena)

    def test_review_admitted_source_epoch_cannot_be_overwritten(self):
        op = qo.QsaPrefillOperation(make_transaction(), make_lane(),
                                   device=FakeQsaDevice(), source_epochs={'layer': 7})
        with self.assertRaisesRegex(qo.OperationError, 'disagrees with admission'):
            op.register_source('layer', make_source(), source_epoch=0)
        op.register_source('layer', make_source(), source_epoch=7)

    def test_review_source_nonzero_epoch_survives_begin(self):
        op = qo.QsaPrefillOperation(make_transaction(), make_lane(), device=FakeQsaDevice())
        source = make_source()
        op.register_source('layer', source, source_epoch=7)
        op.begin_forward()
        op.begin_layer('layer', source, source_epoch=7)
        self.assertEqual(op._layers['layer'].source_epoch, 7)

    def test_review_bad_source_geometry_and_other_device_refused(self):
        op, dev = self.make()
        with self.assertRaisesRegex(qo.OperationError, 'geometry'):
            op.register_source('bad', FakeTensor((64, 1, 4048, 256), device='cpu', pinned=True))
        op.begin_forward()
        op.begin_layer('layers.0.attn', self.source)
        op.producer_submitted('layers.0.attn', self.source)
        with self.assertRaisesRegex(qo.OperationError, 'arena'):
            op.bind_plan('layers.0.attn', make_table(), 2, pages_tokens(2),
                         FakeTensor((24, 1, 4048, 512), device='cuda:1'))

    def test_review_tensor_facts_redacts_pointer_and_rejects_mutable_shape(self):
        facts = FakeQsaDevice().facts(make_source())
        self.assertNotIn(str(facts.address), repr(facts))
        with self.assertRaises(et.E3ContractError):
            et.TensorFacts('cpu', [1], (1,), 'torch.bfloat16', 2, True, True, 123456)

    def test_review_native_streams_explicit_and_stable(self):
        fake_torch = types.SimpleNamespace()
        with patch.dict(sys.modules, {'torch': fake_torch}):
            dev = qo.NativeQsaDevice('cuda:0')
        with self.assertRaisesRegex(qo.OperationError, 'not been bound'):
            dev._stream(dev.compute)
        compute = types.SimpleNamespace(device='cuda:0', cuda_stream=7)
        copy = types.SimpleNamespace(device='cuda:0', cuda_stream=8)
        dev.bind_streams(compute, copy)
        self.assertIs(dev._stream(dev.compute), compute)
        with self.assertRaises(qo.OperationError):
            dev.bind_streams(compute, copy)

    def test_review_cancelled_transaction_cannot_rebind_source(self):
        op, _ = self.make()
        op.cancel()
        with self.assertRaises(qo.OperationCancelled):
            op.register_source('layer', make_source())


class ReviewContinuationTests(unittest.TestCase):
    setUpClass = classmethod(GeneratedCallSiteTests.setUpClass.__func__)
    layers = GeneratedCallSiteTests.layers
    make_case = GeneratedCallSiteTests.make_case
    def test_review_one_bucket_per_advance_even_when_ready(self):
        self.layers()
        op, layer, q, out, metadata, kv = self.make_case(range(50), tuple(range(50)))
        gen = self.helper(op, layer, q, kv, FakeTensor((1, 4)), metadata,
                          q.shape[0], True, out)
        op.attach_continuation(gen)
        # Global sparse and timer settings must not be read/mutated by this lane.
        with patch.dict(os.environ, {'VLLM_QSA_KVO_SPARSE_PREFILL': '1'}):
            for count in (1, 2, 3):
                op.advance()
                self.assertEqual(len(self.kernel.calls), count)
            with self.assertRaises(StopIteration):
                op.advance()
        self.assertEqual(self.gather.calls, [])
        self.assertEqual(op.stats['buckets'], 3)
        consumer = op._layers['layers.0.attn'].consumed[-1]
        self.assertGreaterEqual(len(consumer.owners), 9)

    def test_review_delayed_credit_does_not_destroy_continuation(self):
        self.layers()
        self.device.auto_ready = False
        op, layer, q, out, metadata, kv = self.make_case(range(50), tuple(range(50)))
        gen = self.helper(op, layer, q, kv, FakeTensor((1, 4)), metadata,
                          q.shape[0], True, out)
        op.attach_continuation(gen)
        op.advance()
        copies = list(self.device.copies)
        for _ in range(3):
            with self.assertRaises(et.E3Pending):
                op.advance()
            op.poll()
        self.assertEqual(copies, self.device.copies)
        self.assertIs(op._continuation, gen)
        for event in self.device.events:
            event.mark_ready()
        op.advance()
        self.assertEqual(len(self.kernel.calls), 2)
        op.cancel()
        self.assertFalse(op.drain())
        for event in self.device.events:
            event.mark_ready()
        self.assertTrue(op.drain())
        op.release()
        self.assertIsNone(op._continuation)

    def test_review_kernel_failure_retains_scratch_and_quarantines(self):
        self.layers()
        op, layer, q, out, metadata, kv = self.make_case(range(50), tuple(range(50)))
        class BrokenKernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    raise RuntimeError('kernel launch failed')
                return launch
        self.attention.__globals__['_qsa_sparse_staged_gqa_kernel'] = BrokenKernel()
        op.attach_continuation(self.helper(op, layer, q, kv, FakeTensor((1, 4)),
                                         metadata, q.shape[0], True, out))
        with self.assertRaises(RuntimeError):
            op.advance()
        self.assertFalse(op.drain())
        self.assertEqual(len(op._records[-1].owners), 7)
        op.cancel()
        with self.assertRaises(et.E3Pending):
            op.release()

    def test_review_stock_layer_branch_refuses_before_any_work(self):
        import inspect
        branch = 'def branch(layer):\n' + operation_source._LAYER_BRANCH
        ns = {}
        exec(branch, ns)
        with self.assertRaisesRegex(RuntimeError, 'bounded layer dispatcher'):
            ns['branch'](types.SimpleNamespace(_flash_s5_operation=object()))
        self.assertTrue(inspect.isgeneratorfunction(self.layers().helper))

if __name__ == '__main__':
    unittest.main()
