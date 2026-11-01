"""CPU contract tests for sparse-history DMA transport (no torch, no CUDA).

The planner is cross-checked against the independent VM<vm-id> serial oracle over
all twelve boundary cases.  The source-hold state machine is exercised directly.
The *generated* sparse attention function and layer helper from the pinned E2
composition are executed against the real ``QsaPrefillOperation`` with fake
device/tensor/event objects only at the device boundary.  Recorded copies are
32-byte row signatures, so these tests prove compact order, mapping, bucket
accounting and lifecycle; they do not prove native H2D transport or bitwise
QSA output equality, which require the pinned GPU kernel.
"""
import logging
import ast
import dataclasses
from pathlib import Path
import hashlib
import struct
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
sys.path.insert(0, str(ROOT / 'harness'))

import arena_attention  # noqa: E402
import async_handoff  # noqa: E402
import e3_types as et  # noqa: E402
import operation_source  # noqa: E402
import qsa_operation as qo  # noqa: E402
import runner_hooks  # noqa: E402
import scheduler_hooks  # noqa: E402
import sparse_history_dma as shd  # noqa: E402
import vmid_sparse_plan as vm  # noqa: E402
import wire_v1  # noqa: E402
import worker_bridge  # noqa: E402

LEASE_CATEGORY = async_handoff.LeaseCategory

PAGE_TOKENS = shd.PAGE_ROWS
ROW_WIDTH = 512
HEAD_DIM = ROW_WIDTH // 2
NUM_KV_HEADS = 1
QUERY_HEADS = 6
ARENA_BYTES = shd.ARENA_ROWS * shd.ROW_BYTES

_ITEMS = {'torch.float32': 4, 'torch.int64': 8, 'torch.int32': 4,
          'torch.bfloat16': 2, 'torch.float16': 2, 'torch.bool': 1,
          'torch.uint8': 1}


def _contig(shape):
    strides = []
    step = 1
    for size in reversed(shape):
        strides.append(step)
        step *= size
    return tuple(reversed(strides))


class Rows:
    """Arena row storage: ``Rrows[i]`` is the 32-byte signature written there."""

    def __init__(self, count):
        self.rows = [None] * count

    def clear(self):
        self.rows = [None] * len(self.rows)


class T:
    """Value-carrying fake tensor; only the operations the pinned code uses."""

    _serial = 0x100000

    def __init__(self, shape, *, device='cuda:0', dtype='torch.bfloat16',
                 values=None, strides=None, pinned=False, address=None,
                 storage=None, row_offset=0):
        self.shape = tuple(shape)
        self.device = device
        self.dtype = dtype
        self.strides = tuple(strides) if strides is not None else _contig(self.shape)
        self.values = list(values) if values is not None else None
        self.pinned = pinned
        if address is None:
            T._serial += 0x1000
            address = T._serial
        self.address = address
        self.storage = storage
        self.row_offset = row_offset
        self.item = _ITEMS[dtype]

    # -- geometry -----------------------------------------------------------
    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def numel(self):
        n = 1
        for size in self.shape:
            n *= size
        return n

    def element_size(self):
        return self.item

    def is_pinned(self):
        return self.pinned

    def is_contiguous(self):
        return self.strides == _contig(self.shape)

    def data_ptr(self):
        return self.address

    # -- indexed values -----------------------------------------------------
    def __getitem__(self, key):
        if isinstance(key, T):
            if self.values is None or len(self.shape) != 1:
                raise TypeError('mask index needs a flat value tensor')
            return T((sum(1 for m in key.values if m),), device=self.device,
                     dtype=self.dtype,
                     values=[v for v, m in zip(self.values, key.values) if m])
        if isinstance(key, int):
            if len(self.shape) == 2:
                width = self.shape[1]
                values = None
                if self.values is not None:
                    values = self.values[key * width:(key + 1) * width]
                return T((width,), device=self.device, dtype=self.dtype,
                         values=values, strides=self.strides[1:],
                         address=self.address + key * self.strides[0] * self.item,
                         storage=self.storage,
                         row_offset=self.row_offset + key)
            value = None if self.values is None else self.values[key]
            return T((), device=self.device, dtype=self.dtype,
                     values=None if value is None else [value],
                     address=self.address + key * self.strides[0] * self.item)
        if isinstance(key, tuple):
            rows, cols = key
            if not (rows == slice(None) and isinstance(cols, slice)):
                raise TypeError('fake tensor only supports [:, a:b] views')
            d0, d1 = self.shape
            lo, hi, _ = cols.indices(d1)
            values = None
            if self.values is not None:
                values = [self.values[i * d1 + j]
                          for i in range(d0) for j in range(lo, hi)]
            return T((d0, hi - lo), device=self.device, dtype=self.dtype,
                     values=values, strides=self.strides[:2],
                     address=self.address, storage=self.storage,
                     row_offset=self.row_offset)
        if isinstance(key, slice):
            lo, hi, _ = key.indices(self.shape[0])
            width = self.numel() // self.shape[0] if self.shape else 1
            values = None
            if self.values is not None:
                values = self.values[lo * width:hi * width]
            return T((hi - lo,) + self.shape[1:], device=self.device,
                     dtype=self.dtype, values=values, strides=self.strides,
                     address=self.address + lo * self.strides[0] * self.item,
                     storage=self.storage, row_offset=self.row_offset + lo)
        raise TypeError('unsupported fake tensor view')

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and isinstance(key[0], T):
            rows, cols = key
            lo, hi, _ = cols.indices(self.shape[1])
            width = hi - lo
            for index, row in enumerate(rows.values):
                for j in range(width):
                    if isinstance(value, T):
                        self.values[row * self.shape[1] + lo + j] = \
                            value.values[index * width + j]
                    else:
                        self.values[row * self.shape[1] + lo + j] = value
            return
        if isinstance(key, T):
            width = self.numel() // self.shape[0] if self.shape else 1
            for row in key.values:
                for j in range(width):
                    self.values[row * width + j] = value
            return
        raise TypeError('unsupported fake tensor store')

    # -- unary/binary helpers used by the pinned arithmetic -----------------
    def index_select(self, dim, index):
        if dim != 0 or self.values is None:
            raise TypeError('fake index_select supports dim 0 values only')
        width = self.numel() // self.shape[0]
        values = []
        for row in index.values:
            values.extend(self.values[row * width:(row + 1) * width])
        return T((len(index.values),) + self.shape[1:], device=self.device,
                 dtype=self.dtype, values=values)

    def clone(self):
        return T(self.shape, device=self.device, dtype=self.dtype,
                 values=self.values)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if -1 in shape:
            known = 1
            for size in shape:
                if size != -1:
                    known *= size
            shape = tuple(self.numel() // known if s == -1 else s for s in shape)
        return T(tuple(shape), device=self.device, dtype=self.dtype,
                 values=self.values)

    def to(self, dtype):
        return T(self.shape, device=self.device, dtype=dtype,
                 values=self.values, strides=self.strides, address=self.address)

    def transpose(self, first, second):
        shape = list(self.shape)
        strides = list(self.strides)
        shape[first], shape[second] = shape[second], shape[first]
        strides[first], strides[second] = strides[second], strides[first]
        return T(tuple(shape), device=self.device, dtype=self.dtype,
                 strides=tuple(strides), address=self.address)

    def split(self, size, dim=-1):
        dim %= len(self.shape)
        left = list(self.shape)
        right = list(self.shape)
        left[dim] = size
        right[dim] = self.shape[dim] - size
        return (T(tuple(left), device=self.device, dtype=self.dtype,
                  strides=self.strides, address=self.address),
                T(tuple(right), device=self.device, dtype=self.dtype,
                  strides=self.strides,
                  address=self.address + size * self.strides[dim] * self.item))

    def __ge__(self, other):
        return T(self.shape, device=self.device, dtype='torch.bool',
                 values=[1 if v >= other else 0 for v in self.values])

    def __eq__(self, other):
        return T(self.shape, device=self.device, dtype='torch.bool',
                 values=[1 if v == other else 0 for v in self.values])

    def __isub__(self, other):
        delta = other.values[0] if isinstance(other, T) else other
        self.values = [v - delta for v in self.values]
        return self

    # -- fake transport payload --------------------------------------------
    def row_bytes(self, row):
        return struct.pack('<Q', row) * 4


class FakeTorch:
    Tensor = T
    int64 = 'torch.int64'
    int32 = 'torch.int32'
    float32 = 'torch.float32'
    bool = 'torch.bool'

    def arange(self, start, end, *, device=None, dtype=None):
        return T((int(end) - int(start),), device=str(device),
                 dtype=dtype or 'torch.int64',
                 values=list(range(int(start), int(end))))

    def empty(self, shape, *, dtype=None, device=None):
        return T(tuple(shape), device=str(device),
                 dtype=dtype or 'torch.float32',
                 values=[0] * (shape[0] * shape[1] * shape[2] if len(shape) == 3 else 1))

    def empty_like(self, tensor):
        return T(tensor.shape, device=tensor.device, dtype=tensor.dtype,
                 values=[0] * tensor.numel())

    def unique(self, tensor, *args, **kwargs):
        if tensor.values is None:
            raise TypeError('fake unique needs integer values')
        unique = sorted(set(tensor.values))
        position = {value: i for i, value in enumerate(unique)}
        inverse = [position[value] for value in tensor.values]
        first = T((len(unique),), device=tensor.device, dtype=tensor.dtype,
                  values=unique)
        second = T((len(inverse),), device=tensor.device, dtype='torch.int64',
                   values=inverse)
        return first, second


class FakeTriton:
    def next_power_of_2(self, value):
        return 1 if value <= 1 else 1 << (value - 1).bit_length()

    def cdiv(self, a, b):
        return -(-a // b)


class FakeKernel:
    def __init__(self, name, launch=None):
        self.name = name
        self.calls = []
        self._launch = launch

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            if self._launch is not None:
                self._launch(grid, *args, **kwargs)
        return launch


def union_kernels(block=1024):
    """CPU simulation of the five pinned union kernels.

    The generated ``_s6_union`` wrapper is executed verbatim; only its kernel
    launches are replaced.  Each simulator implements the reviewed kernel's
    index arithmetic over the fake buffers, so the CPU tests exercise the same
    fixed-capacity mark/scan/offsets/pack/remap contract the GPU would run.
    They do not prove native Triton semantics.
    """
    def mark(grid, indices, rows, flags, stride, width, n, count, b):
        for i in range(count * width):
            row = rows.values[i // width]
            value = indices.values[row * stride + i % width]
            if 0 <= value < n:
                flags.values[value] = 1

    def scan(grid, flags, ranks, totals, n, b):
        blocks = grid[0] if isinstance(grid, tuple) else grid
        for program in range(blocks):
            total = 0
            for offset in range(b):
                i = program * b + offset
                if i < n:
                    total += 1 if flags.values[i] else 0
                    ranks.values[i] = total
            totals.values[program] = total

    def offsets(grid, totals, offsets, info, n, groups, b):
        running = 0
        for i in range(b):
            value = totals.values[i] if i < groups else 0
            if i < groups:
                offsets.values[i] = running
            running += value
        info.values[0] = running
        info.values[1] = 1 if running * 5 < n * 4 else 0

    def pack(grid, flags, ranks, offsets, tokens, n, b):
        blocks = grid[0] if isinstance(grid, tuple) else grid
        for program in range(blocks):
            for i in range(program * b, min((program + 1) * b, n)):
                if flags.values[i]:
                    position = ranks.values[i] + offsets.values[program] - 1
                    tokens.values[position] = i
                    ranks.values[i] = position

    def remap(grid, indices, rows, ranks, info, out, stride, out_stride,
              width, count, n, b):
        sparse = info.values[1] != 0
        for i in range(count * width):
            row = rows.values[i // width]
            value = indices.values[row * stride + i % width]
            compact = ranks.values[value] if 0 <= value < n else -1
            out.values[row * out_stride + i % width] = compact if sparse else value

    return {name: FakeKernel(name, launch) for name, launch in (
        ('_s6_mark', mark), ('_s6_scan', scan), ('_s6_offsets', offsets),
        ('_s6_pack', pack), ('_s6_remap', remap))}


class FakeEvent:
    def __init__(self, token, device, ready):
        self.token = token
        self.device = device
        self.ready = ready
        self.queries = 0

    def query(self):
        self.queries += 1
        return self.ready


class FakeSparseDevice:
    """Device boundary only: records copies/events, never synchronizes."""

    compute = 'compute'
    copy = 'copy'

    def __init__(self, *, auto_ready=False):
        self.device = 'cuda:0'
        self.auto_ready = auto_ready
        self.events = []
        self.copies = []
        self.page_copies = []
        self.written = []
        self.index_copies = []
        self.host_reads = []
        self.waits = []
        self.queries = 0
        self.synchronized = 0

    def facts(self, tensor):
        return et.TensorFacts(str(tensor.device), tuple(tensor.shape),
                              tuple(tensor.stride()), str(tensor.dtype),
                              tensor.element_size(), tensor.is_pinned(),
                              tensor.is_contiguous(), tensor.data_ptr())

    def record(self, token):
        event = FakeEvent(token, self.device, self.auto_ready)
        self.events.append(event)
        return event

    def query(self, event):
        self.queries += 1
        return event.query()

    def wait_event(self, token, event):
        self.waits.append((token, event))

    def ready_all(self):
        for event in self.events:
            event.ready = True

    def alloc_union_scratch(self, capacity, rows, width, dtype, remapped_like):
        if not 1 <= capacity <= shd.MAX_LAYER_ROWS:
            raise AssertionError('union scratch bound violated')
        if not 1 <= rows <= shd.MAX_LAYER_ROWS or width < 1:
            raise AssertionError('union geometry bound violated')
        groups = (capacity + 1023) // 1024

        def buffer(shape):
            return T(tuple(shape), device='cuda:0', dtype='torch.int32',
                     values=[0] * shape[0])

        return dict(flags=buffer((capacity,)), ranks=buffer((capacity,)),
                    tokens=buffer((capacity,)), totals=buffer((groups,)),
                    offsets=buffer((groups,)), info=buffer((2,)),
                    remapped=remapped_like.clone())

    def alloc_host_indices(self, capacity, dtype):
        if not 1 <= capacity <= shd.MAX_LAYER_ROWS:
            raise AssertionError('selection scratch bound violated')
        return T((capacity,), device='cpu', dtype=dtype,
                 values=[0] * capacity, pinned=True)

    def copy_indices_to_host(self, source, dest, count):
        if not 1 <= count <= source.numel():
            raise AssertionError('selection readback bound violated')
        for i in range(count):
            dest.values[i] = source.values[i]
        self.index_copies.append((source.address, count))

    def host_indices(self, buffer, count):
        if not buffer.is_pinned() or str(buffer.device) != 'cpu':
            raise AssertionError('selection readback is not pinned CPU')
        self.host_reads.append((buffer.address, count))
        return tuple(buffer.values[:count])

    def copy_rows(self, source, arena, source_row, arena_row, rows):
        if not 1 <= rows <= shd.MAX_PIECE_ROWS:
            raise AssertionError('row copy exceeds the four-page bound')
        storage = arena.storage
        if storage is None:
            raise AssertionError('arena has no row storage')
        for offset in range(rows):
            payload = source.row_bytes(source_row + offset)
            storage.rows[arena.row_offset + arena_row + offset] = payload
            self.written.append(payload)
        self.copies.append((source_row, arena_row, rows, arena.row_offset))

    def copy_pages(self, source, arena, source_page, arena_page, pages):
        self.page_copies.append((source_page, arena_page, pages))

    def synchronize(self):
        self.synchronized += 1
        raise AssertionError('the operation must never synchronize')


class FakeAllocator:
    """Retains a reference object; a Boolean or a lost lease is refused."""

    def __init__(self, *, retain_object=True):
        self.retain_object = retain_object
        self.refs = {}
        self.released = []

    def retain(self, owner):
        if not self.retain_object:
            return True
        reference = object()
        self.refs[id(reference)] = owner
        return reference

    def assert_retained(self, reference):
        if id(reference) not in self.refs:
            raise RuntimeError('allocator lease is not retained')

    def release(self, reference):
        self.released.append(reference)
        self.refs.pop(id(reference), None)


def make_transaction(**overrides):
    values = dict(engine_epoch=1, request_uid='req-sparse', request_generation=0,
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


def make_owner(lane, layer, mapping, valid_tokens, *, transaction=None):
    return shd.SourceOwner(
        transaction or make_transaction(chunk_end=valid_tokens), lane, layer,
        lane.storage_epoch, 0, source_incarnation=1, allocation_incarnation=1,
        physical_pages=tuple(mapping), valid_tokens=valid_tokens,
        mapping_version=1)


class FakeTensorFactsTests(unittest.TestCase):
    def test_fake_tensor_views_keep_row_major_values_and_offsets(self):
        t = T((2, 3), dtype='torch.int32', values=[0, 1, 2, 3, 4, 5])
        view = t[:, :-1]
        self.assertEqual(view.shape, (2, 2))
        self.assertEqual(view.values, [0, 1, 3, 4])
        self.assertEqual(view.data_ptr(), t.data_ptr())
        flat = view.reshape(-1)
        self.assertEqual(flat.values, [0, 1, 3, 4])
        self.assertEqual(t[1:].values, [3, 4, 5])
        self.assertEqual(t[1:].data_ptr(), t.data_ptr() + 3 * 4)


# --- planner cross-check ----------------------------------------------------

def _planner_versions(mapping):
    return tuple(range(1, max(mapping) + 3))


class PlannerTests(unittest.TestCase):
    def test_boundary_fixtures_match_the_vmid_oracle(self):
        self.assertEqual(shd.boundary_cases(), vm.cases())

    def test_every_case_matches_the_independent_oracle(self):
        for name, selected, mapping, fallback in vm.cases():
            versions = _planner_versions(mapping)
            want = vm.make_plan(selected, mapping, versions, 1)
            got = shd.make_sparse_plan(selected, mapping, versions, 1)
            self.assertEqual(got.refused, want.fallback is not None, name)
            if want.fallback:
                self.assertEqual(got.fallback.reason, want.fallback, name)
                self.assertEqual(got.buckets, ())
                continue
            self.assertEqual(
                [(b.compact_start, b.rows,
                  [(p.source_row, p.arena_row, p.rows) for p in b.pieces])
                 for b in got.buckets],
                [(b.compact_start, b.rows,
                  [(p.source_row, p.arena_row, p.rows) for p in b.pieces])
                 for b in want.buckets], name)

    def test_piece_bucket_and_descriptor_bounds(self):
        for name, selected, mapping, _ in vm.cases():
            plan = shd.make_sparse_plan(selected, mapping,
                                        _planner_versions(mapping), 1)
            if plan.refused:
                continue
            cursor = 0
            for bucket in plan.buckets:
                self.assertLessEqual(bucket.rows, shd.ARENA_ROWS)
                self.assertLessEqual(bucket.descriptors, shd.MAX_DESCRIPTORS)
                self.assertEqual(bucket.compact_start, cursor)
                cursor = bucket.compact_end
                arena_row = 0
                for piece in bucket.pieces:
                    self.assertEqual(piece.arena_row, arena_row)
                    arena_row = piece.arena_end
                    self.assertLessEqual(piece.rows, shd.MAX_PIECE_ROWS)
                    self.assertLessEqual(piece.rows * shd.ROW_BYTES,
                                         shd.MAX_PIECE_PAGES * et.PAGE_BYTES)
                    self.assertLessEqual(piece.source_pages,
                                         shd.MAX_PIECE_PAGES)
                self.assertEqual(arena_row, bucket.rows)
            self.assertEqual(cursor, len(selected))

    def test_descriptor_cap_and_null_page_are_typed_refusals(self):
        selected = tuple(range(0, 1024, 2))
        mapping = tuple((i * 17 + 3) % 52 for i in range(50))
        versions = _planner_versions(mapping)
        plan = shd.make_sparse_plan(selected, mapping, versions, 1)
        self.assertTrue(plan.refused)
        self.assertEqual(plan.fallback.reason, 'descriptor_cap')
        self.assertEqual(plan.fallback.descriptors, shd.MAX_DESCRIPTORS + 1)
        with self.assertRaises(shd.SparsePlanRefused) as caught:
            shd.validate_plan_source(plan, mapping, versions, 1)
        self.assertEqual(caught.exception.refusal.reason, 'descriptor_cap')
        unsupported = shd.make_sparse_plan((7, 8), (3, 4),
                                           _planner_versions((3, 4)), 1,
                                           null_page=3)
        self.assertTrue(unsupported.refused)
        self.assertEqual(unsupported.fallback.reason, 'unsupported_mapping')
        with self.assertRaises(shd.SparsePlanRefused):
            shd.validate_plan_source(unsupported, (3, 4),
                                     _planner_versions((3, 4)), 1)

    def test_stale_map_versions_and_epoch_are_refused(self):
        mapping = (3, 4, 5)
        versions = (1, 2, 3, 4, 5, 6)
        plan = shd.make_sparse_plan((0, 4049, 8100), mapping, versions, 2)
        self.assertFalse(plan.refused)
        shd.validate_plan_source(plan, mapping, versions, 2)
        with self.assertRaises(shd.SourceRefused):
            shd.validate_plan_source(plan, mapping, (1, 2, 4, 4, 5, 6), 2)
        with self.assertRaises(shd.SourceRefused):
            shd.validate_plan_source(plan, (3, 4, 6), versions, 2)
        with self.assertRaises(shd.SourceRefused):
            shd.validate_plan_source(plan, mapping, versions, 3)
        with self.assertRaises(shd.SparsePlanError):
            shd.make_sparse_plan((0, 0), mapping, versions, 2)
        with self.assertRaises(shd.SparsePlanError):
            shd.make_sparse_plan((0, 4049), mapping, versions, 2,
                                 valid_tokens=4048)

    def test_piece_merge_needs_source_and_destination_adjacency(self):
        mapping = (5, 9, 10)
        versions = _planner_versions(mapping)
        contiguous = shd.make_sparse_plan(tuple(range(3 * PAGE_TOKENS)),
                                          mapping, versions, 1)
        self.assertEqual([p.rows for p in contiguous.buckets[0].pieces],
                         [PAGE_TOKENS, 2 * PAGE_TOKENS])
        aliased = shd.make_sparse_plan((0, 1, PAGE_TOKENS, PAGE_TOKENS + 1),
                                       (3, 3), _planner_versions((3, 3)), 1)
        self.assertEqual([(p.source_row, p.rows)
                          for p in aliased.buckets[0].pieces],
                         [(3 * PAGE_TOKENS, 2), (3 * PAGE_TOKENS, 2)])


# --- source hold ------------------------------------------------------------

class SourceHoldTests(unittest.TestCase):
    def setUp(self):
        self.lane = make_lane()
        self.mapping = (3, 7, 9)
        self.hold = shd.SparseSourceHold(
            make_owner(self.lane, 'qsa.0', self.mapping, 3 * PAGE_TOKENS),
            allocator=FakeAllocator())

    def test_validate_proves_identity_incarnations_map_and_lease(self):
        owner = self.hold.owner
        self.assertEqual(self.hold.validate(owner, 'qsa.0', self.mapping),
                         self.mapping)
        for field, value in (
                ('transaction', make_transaction(txn_seq=2)),
                ('lane', make_lane(worker_rank=1)),
                ('storage_epoch', 9),
                ('source_epoch', 4),
                ('source_incarnation', 2),
                ('allocation_incarnation', 5),
                ('valid_tokens', 3 * PAGE_TOKENS - 1),
                ('mapping_version', 2)):
            stale = shd.SourceOwner(
                value if field == 'transaction' else owner.transaction,
                value if field == 'lane' else owner.lane, owner.layer,
                value if field == 'storage_epoch' else owner.storage_epoch,
                value if field == 'source_epoch' else owner.source_epoch,
                value if field == 'source_incarnation' else owner.source_incarnation,
                value if field == 'allocation_incarnation' else owner.allocation_incarnation,
                owner.physical_pages,
                value if field == 'valid_tokens' else owner.valid_tokens,
                value if field == 'mapping_version' else owner.mapping_version)
            with self.assertRaises(shd.SourceRefused, msg=field):
                self.hold.validate(stale, 'qsa.0', self.mapping)
        with self.assertRaises(shd.SourceRefused):
            self.hold.validate(owner, 'qsa.1', self.mapping)
        with self.assertRaises(shd.SourceRefused):
            self.hold.validate(owner, 'qsa.0', (3, 7, 8))
        other = shd.SparseSourceHold(
            make_owner(self.lane, 'qsa.0', self.mapping, 3 * PAGE_TOKENS),
            allocator=FakeAllocator())
        with self.assertRaises(shd.SourceRefused):
            self.hold.validate(other.owner, 'qsa.0', self.mapping)

    def test_a_boolean_pin_is_not_a_retained_reference(self):
        with self.assertRaises(shd.SourceRefused):
            shd.SparseSourceHold(
                make_owner(self.lane, 'qsa.0', self.mapping, 3 * PAGE_TOKENS),
                allocator=FakeAllocator(retain_object=False))
        with self.assertRaises(shd.SourceRefused):
            shd.SparseSourceHold(self.hold.owner, allocator=object())

    def test_source_is_held_until_copy_and_consumer_complete(self):
        self.hold.begin_bucket(0, 8, 2, 8 * shd.ROW_BYTES)
        with self.assertRaises(shd.E3Pending):
            self.hold.release()
        with self.assertRaises(shd.SourceRefused):
            self.hold.publish(0)
        self.hold.copy_recorded(0)
        self.hold.publish(0)
        with self.assertRaises(shd.E3Pending):
            self.hold.release()
        self.hold.copy_completed(0)
        with self.assertRaises(shd.E3Pending):
            self.hold.release()
        self.hold.consumer_completed(0)
        info = self.hold.release()
        self.assertTrue(info['released'])
        self.assertEqual(info['consumed'], 1)
        with self.assertRaises(shd.SourceRefused):
            self.hold.validate(self.hold.owner, 'qsa.0', self.mapping)
        with self.assertRaises(shd.SourceRefused):
            self.hold.release()

    def test_partial_bucket_cannot_be_published_abandoned_or_reused(self):
        self.hold.begin_bucket(0, 8, 2, 8 * shd.ROW_BYTES)
        with self.assertRaises(shd.SourceRefused):
            self.hold.abandon(0)          # copies not recorded/drained
        with self.assertRaises(shd.SourceRefused):
            self.hold.begin_bucket(0, 8, 2, 8 * shd.ROW_BYTES)
        self.hold.copy_recorded(0)
        with self.assertRaises(shd.SourceRefused):
            self.hold.abandon(0)          # the copy event was not observed
        self.hold.copy_completed(0)
        self.hold.abandon(0)
        self.assertTrue(self.hold.releasable)
        with self.assertRaises(shd.SourceRefused):
            self.hold.consumer_completed(0)   # an abandoned bucket stays dead
        self.hold.release()
        with self.assertRaises(shd.SourceRefused):
            self.hold.begin_bucket(1, 8, 2, 8 * shd.ROW_BYTES)

    def test_published_bucket_may_drain_and_be_abandoned_after_cancel(self):
        self.hold.begin_bucket(0, 4, 1, 4 * shd.ROW_BYTES)
        self.hold.copy_recorded(0)
        self.hold.publish(0)
        self.hold.copy_completed(0)
        self.hold.abandon(0)
        self.assertTrue(self.hold.releasable)
        self.assertEqual(self.hold.release()['abandoned'], 1)

    def test_bucket_accounting_bounds(self):
        for args in ((0, 0, 1, 0), (0, 1, 0, shd.ROW_BYTES),
                     (0, 1, shd.MAX_DESCRIPTORS + 1, shd.ROW_BYTES),
                     (0, 1, 1, shd.ROW_BYTES + 1)):
            with self.assertRaises(shd.SourceRefused):
                self.hold.begin_bucket(*args)


# --- generated sparse call sites -------------------------------------------

def _function_node(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError('generated function missing: ' + name)


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


REFERENCE = ROOT / 'reference/vllm-src/models/qwen4_exp/nvidia'


class _GeneratedHarness:
    @classmethod
    def setUpClass(cls):
        if not REFERENCE.is_dir():
            raise unittest.SkipTest('reference/vllm-src is not available on this host')
        cls.ops_e2 = arena_attention.generate((REFERENCE / 'ops/qsa.py').read_text())
        cls.qsa_e2 = runner_hooks.qsa_staged((REFERENCE / 'qsa.py').read_text())
        cls.generated_ops = operation_source.generate_ops(cls.ops_e2)
        cls.generated_qsa = operation_source.generate_layer(cls.qsa_e2)

    # -- fixtures -----------------------------------------------------------
    def setUp(self):
        self.device = FakeSparseDevice(auto_ready=True)
        self.torch = FakeTorch()
        self.union_kernels = union_kernels()
        self.kernel = FakeKernel('gqa')
        self.arena = T((shd.ARENA_PAGES, NUM_KV_HEADS, PAGE_TOKENS, ROW_WIDTH),
                       device='cuda:0', dtype='torch.bfloat16',
                       storage=Rows(shd.ARENA_ROWS))
        self.arena.address = 0x900000
        self.table = T((260,), device='cuda:0', dtype='torch.int64',
                       values=[0] * 260)
        self.attention = self.compile_attention()
        self.helper = self.compile_helper()

    def compile_attention(self):
        module = ast.Module(
            body=[_function_node(self.generated_ops, 'flash_s5_sparse_arena_attention'),
                  _function_node(self.generated_ops, '_s6_union')],
            type_ignores=[])
        ast.fix_missing_locations(module)
        globals_ops = {
            '__name__': 'ops.qsa',
            'torch': self.torch,
            'os': __import__('os'),
            'triton': FakeTriton(),
            '_STAGE_TIMER': None,
            '_qsa_sparse_staged_gqa_kernel': self.kernel,
            '_qsa_gather_selected_host_rows': FakeKernel('gather'),
            '_select_config': lambda rows, heads, prefill, width: (64, 4, 1, None),
        }
        globals_ops.update(self.union_kernels)
        exec(compile(module, '<generated sparse ops>', 'exec'), globals_ops)
        self.union_runner = globals_ops['_s6_union']
        return globals_ops['flash_s5_sparse_arena_attention']

    def compile_helper(self):
        package_name = 'models.qwen4_exp.nvidia.ops.qsa'
        package = _module(package_name,
                          flash_s5_sparse_arena_attention=self.attention,
                          flash_s5_arena_attention=self.attention,
                          flash_e2_arena_attention=lambda *args, **kwargs: None,
                          qsa_get_staging_arena=lambda kv_cache, nbytes: self.arena)
        for name in ('models', 'models.qwen4_exp', 'models.qwen4_exp.nvidia',
                     'models.qwen4_exp.nvidia.ops'):
            sys.modules.setdefault(name, _module(name))
        sys.modules[package_name] = package
        self.addCleanup(sys.modules.pop, package_name, None)
        module = ast.Module(
            body=[_function_node(self.generated_qsa,
                                 '_flash_s5_sparse_staged_operation')],
            type_ignores=[])
        ast.fix_missing_locations(module)
        globals_helper = {'__name__': 'models.qwen4_exp.nvidia.qsa',
                          '__package__': 'models.qwen4_exp.nvidia',
                          'torch': self.torch}
        exec(compile(module, '<generated sparse qsa>', 'exec'), globals_helper)
        return globals_helper['_flash_s5_sparse_staged_operation']

    def make_source(self, pages):
        return T((pages, NUM_KV_HEADS, PAGE_TOKENS, ROW_WIDTH), device='cpu',
                 dtype='torch.bfloat16', pinned=True)

    def run_generator(self, operation, generator, *, auto_mark=True):
        operation.attach_continuation(generator)
        selections = 0
        buckets = 0
        for _ in range(200):
            try:
                value = operation.advance()
            except StopIteration:
                return selections, buckets
            except et.E3Pending:
                if not auto_mark:
                    raise
                self.device.ready_all()
                continue
            if getattr(value, 'qsa_schedule_boundary', False):
                if auto_mark:
                    self.device.ready_all()
                self.authorize_unit_profile(operation)
                continue
            if getattr(value, 'sparse_selection_pending', False):
                selections += 1
                if auto_mark:
                    self.device.ready_all()
                continue
            buckets += 1
            if auto_mark:
                self.device.ready_all()
        self.fail('sparse generator did not terminate')

    def authorize_unit_profile(self, operation):
        """Single-rank component-test coordinator, not engine/TP evidence.

        The queried real local plan supplies the shape. The synthetic cohort
        replaces EngineCore's real agreement RPC and is more convenient; the
        separate native18 connected suite exercises active-stage agreement.
        """
        shape = operation.schedule_profile()
        if shape is None:
            raise et.E3Pending('component profile selection pending')
        query = wire_v1.QsaProfileQuery(operation.transaction, 1, 0, 0, 0,
                                        (operation.lane.worker_rank,))
        grant = wire_v1.QsaProfileGrant(query, shape)
        operation.authorize_schedule(grant)
        return grant


class GeneratedSparseTests(_GeneratedHarness, unittest.TestCase):
    def make_operation(self, layer_tables, versions=None):
        lane = make_lane()
        transaction = make_transaction(chunk_end=1 << 24)
        operation = qo.QsaPrefillOperation(
            transaction, lane, device=self.device,
            plan_provider=qo.metadata_plan_provider(
                Snapshot([Row()], range(260)),
                layer_tables=tuple(layer_tables)))
        # The admitted state epoch is an explicit part of the source identity;
        # these fixtures construct the operation directly instead of through
        # the bridge, so they declare the frame themselves.
        operation.state_epoch = 1
        return operation, lane, transaction

    def hold_for(self, lane, transaction, layer, mapping, tokens):
        return shd.SparseSourceHold(
            make_owner(lane, layer, mapping, tokens, transaction=transaction),
            allocator=FakeAllocator())

    # -- execution helpers --------------------------------------------------
    def bind(self, operation, lane, transaction, layer, mapping, tokens):
        pool = max(max(mapping) + 1, (tokens + PAGE_TOKENS - 1) // PAGE_TOKENS)
        source = self.make_source(pool)
        versions = tuple(range(1, pool + 3))
        operation.register_source(layer, source, versions=versions)
        return source

    def prepare(self, mappings):
        """Register every layer's pinned source, then open the forward."""
        operation, lane, transaction = self.make_operation(list(mappings.items()))
        sources = {}
        for name, mapping in mappings.items():
            sources[name] = self.bind(operation, lane, transaction, name,
                                      mapping, len(mapping) * PAGE_TOKENS)
        operation.begin_forward(layers=list(mappings))
        return operation, lane, transaction, sources

    def open_layer(self, operation, lane, transaction, name, source, mapping,
                   min_pages=1):
        tokens = len(mapping) * PAGE_TOKENS
        operation.begin_layer(name, source)
        operation.producer_submitted(name, source)
        operation.bind_plan(name, self.table, len(mapping), tokens, self.arena)
        hold = self.hold_for(lane, transaction, name, mapping, tokens)
        operation.attach_sparse_source(name, hold)
        operation.bind_sparse_context(name, self.arena, self.table,
                                      min_pages=min_pages)
        return hold

    def expected_rows(self, selected, mapping):
        return [mapping[token // PAGE_TOKENS] * PAGE_TOKENS + token % PAGE_TOKENS
                for token in selected]

    def arena_rows(self, count):
        return self.arena.storage.rows[:count]

    def arena_digest(self, count):
        digest = hashlib.sha256()
        rows = self.arena.storage.rows[:count]
        for row in rows:
            digest.update(b'-' if row is None else row)
        return len(rows), digest.hexdigest()

    def expected_digest(self, selected, mapping):
        digest = hashlib.sha256()
        for row in self.expected_rows(selected, mapping):
            digest.update(struct.pack('<Q', row) * 4)
        return len(selected), digest.hexdigest()

    def sparse_case(self, *, pages=3, mapping=(3, 7, 9), width=5,
                    rows_values=None, tokens=None, layers=1):
        tokens = tokens or pages * PAGE_TOKENS
        names = [f'qsa.{i}' for i in range(layers)]
        operation, lane, transaction, sources = self.prepare(
            {name: mapping for name in names})
        nrows = 8 if rows_values is None else len(rows_values)
        values = rows_values or [
            [(r * width + c) for c in range(width - 1)] for r in range(nrows)]
        flat = []
        for row in values:
            flat.extend(row)
            flat.extend([-1] * (width - len(row)))
        logical = T((nrows, width), device='cuda:0', dtype='torch.int32',
                    values=flat)
        query = T((nrows, QUERY_HEADS, HEAD_DIM), device='cuda:0',
                  dtype='torch.bfloat16', values=[0.0] * (nrows * QUERY_HEADS * HEAD_DIM))
        output = T((nrows, QUERY_HEADS, HEAD_DIM), device='cuda:0',
                   dtype='torch.bfloat16', values=[0.0] * (nrows * QUERY_HEADS * HEAD_DIM))
        rows = T((nrows,), device='cuda:0', dtype='torch.int64',
                 values=list(range(nrows)))
        buckets = 0
        for name in names:
            self.arena.storage.clear()
            self.open_layer(operation, lane, transaction, name,
                            sources[name], mapping)
            generator = self.attention(
                query, sources[name], logical, self.table, rows, len(mapping) - 1,
                self.arena, True, output, e3op=operation)
            selections, buckets = self.run_generator(operation, generator)
            self.assertEqual(selections, 0)
            operation.end_layer(name)
        operation.end_forward()
        self.device.ready_all()
        self.assertTrue(operation.drain())
        return operation, selections, buckets

    # -- tests --------------------------------------------------------------
    def test_generated_sparse_overlay_is_executable_and_arithmetic_preserved(self):
        source = self.generated_ops
        sparse = source[source.index('\n\ndef flash_s5_sparse_arena_attention('):]
        self.assertIn('def flash_s5_sparse_arena_attention(', source)
        # The host-synchronising dynamic union is gone; the fixed-capacity
        # mark/scan/pack/remap overlay is appended with only the five pinned
        # kernels (the old UVA gather is deliberately absent).
        self.assertNotIn('torch.unique', sparse)
        self.assertNotIn('unique[', sparse)
        self.assertNotIn('int(unique[0])', sparse)
        self.assertNotIn('unique_valid', sparse)
        self.assertIn('e3op.sparse_union_scratch(capacity, logical_indices, rows)', sparse)
        self.assertIn('_s6_union(', sparse)
        self.assertIn('e3op.stage_sparse_selection(unique_tokens, union_info)', sparse)
        self.assertIn('while not e3op.sparse_selection_ready(_selection):', sparse)
        self.assertIn('e3op.sparse_selection_pending(_selection)', sparse)
        self.assertIn('e3op.sparse_selection_count(_selection)', sparse)
        self.assertIn('e3op.fill_sparse_bucket(', sparse)
        self.assertIn('selected_count * 5 < capacity * 4', sparse)
        self.assertIn('def _s6_union(', source)
        self.assertNotIn('_s6_gather', source)
        self.assertEqual(source.count('def _s6_mark('), 1)
        self.assertEqual(source.count('def _s6_remap('), 1)
        self.assertEqual(operation_source.revert_ops(source), self.ops_e2)
        self.assertEqual(operation_source.revert_layer(self.generated_qsa),
                         self.qsa_e2)
        self.assertIn('operation.bind_sparse_context(', self.generated_qsa)

    def test_compact_rows_and_remap_drive_the_arena_exactly(self):
        self.arena.storage.clear()
        operation, selections, buckets = self.sparse_case()
        self.assertEqual(selections, 0)
        self.assertEqual(buckets, 1)
        run = operation._layers['qsa.0']
        plan = run.sparse.plan
        self.assertFalse(plan.refused)
        expected = self.expected_rows(plan.selected, (3, 7, 9))
        self.assertEqual(self.arena_rows(len(expected)),
                         [struct.pack('<Q', row) * 4 for row in expected])
        self.assertIsNone(self.arena.storage.rows[len(expected)])
        # The kernel receives compact ordinals, not the original token ids.
        kernel_args = self.kernel.calls[0][1]
        remapped = kernel_args[3].values
        width = 5
        self.assertEqual(remapped.count(-1), len(remapped) // width)
        self.assertTrue(all(0 <= value < len(expected)
                            for index, value in enumerate(remapped)
                            if index % width != width - 1))
        # The async readback copies the whole fixed-capacity token vector and
        # the two-word info record; the host then uses only the count prefix.
        capacity = 3 * PAGE_TOKENS
        self.assertEqual([count for _, count in self.device.index_copies],
                         [capacity, 2])
        self.assertEqual(operation.stats['selection_bytes'],
                         capacity * 4 + 2 * 4)
        self.assertEqual(operation.stats['sparse_bytes'],
                         len(expected) * shd.ROW_BYTES)
        self.assertEqual(operation.stats['selection_copies'], 1)
        self.assertEqual(self.device.synchronized, 0)
        operation.release()

    def test_fixed_capacity_union_matches_the_independent_oracle(self):
        # Duplicates, -1 padding, out-of-range entries and page edges all go
        # through the generated fixed-capacity wrapper.  The trailing per-row
        # count column must survive the clone-and-remap exactly.
        nrows, width, capacity = 4, 4, 4096
        rows_values = [
            [0, 4047, 9, -1],
            [4048, 9, 9000, -1],
            [4095, 0, 4048, -1],
            [-1, -1, -1, -1],
        ]
        flat = []
        for row in rows_values:
            flat.extend(row)
            flat.append(4242)
        logical = T((nrows, width + 1), device='cuda:0', dtype='torch.int32',
                    values=flat)
        rows = T((nrows,), device='cuda:0', dtype='torch.int64',
                 values=list(range(nrows)))
        buffers = self.device.alloc_union_scratch(capacity, nrows, width,
                                                  'torch.int32', logical)
        tokens, info, remapped = self.union_runner(
            logical, rows, capacity, types.SimpleNamespace(**buffers))
        oracle = sorted({value for row in rows_values for value in row
                         if 0 <= value < capacity})
        self.assertEqual(oracle, [0, 9, 4047, 4048, 4095])
        self.assertEqual(info.values, [len(oracle), 1])
        self.assertEqual(tokens.values[:len(oracle)], oracle)
        compact = {value: index for index, value in enumerate(oracle)}
        for index, row in enumerate(rows_values):
            for column, value in enumerate(row):
                self.assertEqual(
                    remapped.values[index * (width + 1) + column],
                    compact.get(value, -1))
            self.assertEqual(remapped.values[index * (width + 1) + width], 4242)
        for name, kernel in self.union_kernels.items():
            self.assertEqual(len(kernel.calls), 1, name)
        self.assertEqual(self.device.synchronized, 0)

    def test_dense_route_flag_keeps_the_original_inverse_bytes(self):
        # A fully selected capacity takes the pinned dense branch: the device
        # flag is 0 and the remap writes the original values back unchanged.
        nrows, width, capacity = 2, 4, 8
        rows_values = [list(range(4)), list(range(4, 8))]
        flat = []
        for row in rows_values:
            flat.extend(row)
            flat.append(7)
        logical = T((nrows, width + 1), device='cuda:0', dtype='torch.int32',
                    values=flat)
        rows = T((nrows,), device='cuda:0', dtype='torch.int64',
                 values=list(range(nrows)))
        buffers = self.device.alloc_union_scratch(capacity, nrows, width,
                                                  'torch.int32', logical)
        tokens, info, remapped = self.union_runner(
            logical, rows, capacity, types.SimpleNamespace(**buffers))
        self.assertEqual(info.values, [capacity, 0])
        self.assertEqual(tokens.values[:capacity], list(range(capacity)))
        self.assertEqual(remapped.values, flat)

    def test_duplicate_and_aliased_pages_collapse_to_unique_logical_rows(self):
        mapping = (3, 3)                 # two logical pages alias one physical
        rows_values = [[0, 4047], [4048, 8095]]   # page edges on both sides
        operation, _, _ = self.sparse_case(pages=2, mapping=mapping, width=3,
                                           rows_values=rows_values)
        run = operation._layers['qsa.0']
        plan = run.sparse.plan
        self.assertEqual(plan.selected, (0, 4047, 4048, 8095))
        expected = self.expected_rows(plan.selected, mapping)
        self.assertEqual(expected, [3 * PAGE_TOKENS, 3 * PAGE_TOKENS + 4047,
                                    3 * PAGE_TOKENS, 3 * PAGE_TOKENS + 4047])
        self.assertEqual(self.arena_rows(len(expected)),
                         [struct.pack('<Q', row) * 4 for row in expected])
        self.assertEqual(len(self.device.written), 4)
        self.assertEqual(self.device.written[0], self.device.written[2])
        self.assertEqual(self.device.written[1], self.device.written[3])
        remapped = self.kernel.calls[0][1][3].values
        self.assertEqual(remapped[:2], [0, 1])
        self.assertEqual(remapped[3:5], [2, 3])
        operation.release()

    def test_nearly_dense_selection_keeps_the_pinned_dense_transfer(self):
        mapping = (3, 7, 9)
        nrows, width = 3, PAGE_TOKENS + 1
        flat = []
        for row in range(nrows):
            flat.extend(range(row * PAGE_TOKENS, (row + 1) * PAGE_TOKENS))
            flat.append(-1)
        operation, lane, transaction, sources = self.prepare({'qsa.0': mapping})
        hold = self.open_layer(operation, lane, transaction, 'qsa.0',
                              sources['qsa.0'], mapping)
        logical = T((nrows, width), device='cuda:0', dtype='torch.int32',
                    values=flat)
        query = T((nrows, QUERY_HEADS, HEAD_DIM), device='cuda:0',
                  dtype='torch.bfloat16')
        output = T((nrows, QUERY_HEADS, HEAD_DIM), device='cuda:0',
                   dtype='torch.bfloat16')
        rows = T((nrows,), device='cuda:0', dtype='torch.int64',
                 values=list(range(nrows)))
        generator = self.attention(query, sources['qsa.0'], logical,
                                   self.table, rows, len(mapping) - 1,
                                   self.arena, True, output, e3op=operation)
        selections, buckets = self.run_generator(operation, generator)
        self.assertEqual((selections, buckets), (0, 1))
        run = operation._layers['qsa.0']
        self.assertIsNone(run.sparse.plan)          # dense: no CPU row plan
        self.assertEqual(run.sparse.selection_count, nrows * PAGE_TOKENS)
        self.assertFalse(run.sparse.selection_sparse)
        self.assertEqual(operation.stats['sparse_routes'], 1)
        self.assertEqual(self.device.copies, [])
        self.assertEqual(self.device.page_copies,
                         [(3, 0, 1), (7, 1, 1), (9, 2, 1)])
        # The attention kernel sees the original selected indices, not a
        # compact remap, and the ordinary dense attention rows are preserved.
        self.assertEqual(self.kernel.calls[0][1][3].values, flat)
        operation.end_layer('qsa.0')
        operation.end_forward()
        self.device.ready_all()
        self.assertTrue(operation.drain())
        operation.release()
        self.assertTrue(hold.released)

    def test_selection_readback_is_a_separate_resumed_phase(self):
        self.device.auto_ready = False
        mapping = (3, 7, 9)
        operation, lane, transaction, sources = self.prepare({'qsa.0': mapping})
        source = sources['qsa.0']
        hold = self.open_layer(operation, lane, transaction, 'qsa.0', source,
                               mapping)
        logical = T((4, 4), device='cuda:0', dtype='torch.int32',
                    values=[0, 4090, 50, -1, 7, 4090, 50, -1,
                            9, 4090, 50, -1, 11, 4090, 50, -1])
        query = T((4, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
        output = T((4, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
        rows = T((4,), device='cuda:0', dtype='torch.int64', values=[0, 1, 2, 3])
        generator = self.attention(query, source, logical, self.table, rows, 2,
                                   self.arena, True, output, e3op=operation)
        operation.attach_continuation(generator)
        first = operation.advance()
        self.assertTrue(first.qsa_schedule_boundary)
        self.assertEqual(self.device.copies, [])
        self.assertEqual(operation.stats['selection_copies'], 1)
        # Nonblocking submit: the generator yielded while the copy event was
        # pending.  Nothing was synchronised and no pinned byte was read.
        token = operation._layers['qsa.0'].sparse.selection_token
        self.assertEqual(operation.pending_selection, 1)
        self.assertFalse(operation.sparse_selection_ready(token))
        self.assertEqual(self.device.host_reads, [])
        self.assertEqual(self.device.synchronized, 0)
        self.assertGreaterEqual(self.device.queries, 1)
        with self.assertRaises(qo.OperationError):
            operation.sparse_selection_count(token)
        with self.assertRaises(et.E3Pending):
            operation.advance()          # no new accepted boundary while waiting
        self.assertIsNone(operation.schedule_profile())
        self.assertEqual(operation.stats['selection_copies'], 1)
        self.assertEqual(self.device.host_reads, [])
        self.assertIsNone(operation._layers['qsa.0'].sparse.plan)
        self.device.ready_all()
        self.authorize_unit_profile(operation)
        operand = operation.advance()
        self.assertFalse(getattr(operand, 'sparse_selection_pending', False))
        plan = operation._layers['qsa.0'].sparse.plan
        self.assertIsNotNone(plan)
        self.assertEqual([count for _, count in self.device.host_reads],
                         [2, len(plan.selected)])
        self.assertEqual(len(self.device.copies),
                         len(operation._layers['qsa.0'].sparse.plan.buckets[0].pieces))
        pending = False
        try:
            operation.advance()           # consumer not queried yet
        except et.E3Pending:
            pending = True
        self.assertTrue(pending)
        self.device.ready_all()
        try:
            operation.advance()
        except StopIteration:
            pass
        else:
            self.fail('sparse continuation did not finish')
        operation.end_layer('qsa.0')
        operation.end_forward()
        self.device.ready_all()
        self.assertTrue(operation.drain())
        operation.release()

    def test_two_buckets_follow_the_pinned_bucket_boundary(self):
        self.arena.storage.clear()
        pages = 124
        tokens = pages * PAGE_TOKENS
        nrows, width = 60, 2052
        rows_values = [[r * (width - 1) + c for c in range(width - 1)]
                       for r in range(nrows)]
        operation, selections, buckets = self.sparse_case(
            pages=pages, mapping=tuple(range(pages)), width=width,
            rows_values=rows_values, tokens=tokens)
        self.assertEqual(selections, 0)
        self.assertEqual(buckets, 2)
        run = operation._layers['qsa.0']
        plan = run.sparse.plan
        self.assertEqual(plan.bucket_rows, (shd.ARENA_ROWS,
                                            nrows * (width - 1) - shd.ARENA_ROWS))
        digest = hashlib.sha256()
        for payload in self.device.written:
            digest.update(payload)
        self.assertEqual((len(self.device.written), digest.hexdigest()),
                         self.expected_digest(plan.selected,
                                              tuple(range(pages))))
        flags = [(kwargs['FIRST_BUCKET'], kwargs['LAST_BUCKET'])
                 for _, _, kwargs in self.kernel.calls]
        self.assertEqual(flags, [(True, False), (False, True)])
        self.assertEqual(self.device.copies,
                         [(p.source_row, p.arena_row, p.rows, 0)
                          for b in plan.buckets for p in b.pieces])
        self.assertEqual(operation.stats['sparse_buckets'], 2)
        self.assertEqual(operation.stats['sparse_descriptor_high_water'],
                         plan.descriptor_high_water)
        operation.release()

    def test_fragmented_mapping_is_recorded_piece_by_piece(self):
        mapping = (9, 4, 12)
        rows_values = [[0, 4050, 8099, 4049], [1, 4051, 8100, 4048]]
        operation, _, _ = self.sparse_case(mapping=mapping, width=5,
                                           rows_values=rows_values)
        run = operation._layers['qsa.0']
        pieces = [(p.source_row, p.arena_row, p.rows)
                  for p in run.sparse.plan.buckets[0].pieces]
        self.assertGreater(len(pieces), 1)
        self.assertEqual([rows for _, _, rows, _ in self.device.copies],
                         [p[2] for p in pieces])
        expected = self.expected_rows(run.sparse.plan.selected, mapping)
        self.assertEqual(self.arena_rows(len(expected)),
                         [struct.pack('<Q', row) * 4 for row in expected])
        operation.release()

    def test_four_layers_each_stage_their_own_actual_selection(self):
        self.arena.storage.clear()
        mappings = [(3, 7, 9), (4, 8, 10), (5, 9, 11), (6, 10, 12)]
        names = [f'qsa.{i}' for i in range(4)]
        operation, lane, transaction, sources = self.prepare(
            dict(zip(names, mappings)))
        selected_per_layer = []
        for index, name in enumerate(names):
            self.arena.storage.clear()
            self.open_layer(operation, lane, transaction, name,
                            sources[name], mappings[index])
            base = index * 4
            logical = T((2, 4), device='cuda:0', dtype='torch.int32',
                        values=[base, 4090 + index, 50, -1,
                                900 + index, 8100, 51, -1])
            query = T((2, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
            output = T((2, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
            rows = T((2,), device='cuda:0', dtype='torch.int64', values=[0, 1])
            generator = self.attention(query, sources[name], logical,
                                       self.table, rows, 2, self.arena, True,
                                       output, e3op=operation)
            self.run_generator(operation, generator)
            operation.end_layer(name)
            plan = operation._layers[name].sparse.plan
            selected_per_layer.append(plan.selected)
            expected = self.expected_rows(plan.selected, mappings[index])
            self.assertEqual(self.arena_rows(len(expected)),
                             [struct.pack('<Q', row) * 4 for row in expected],
                             name)
        operation.end_forward()
        self.assertEqual(len(set(selected_per_layer)), 4)
        self.assertEqual(operation.stats['sparse_buckets'], 4)
        self.device.ready_all()
        self.assertTrue(operation.drain())
        operation.release()

    # -- S1: union scratch retention (leaves: harness/alpha_s1_fixture_leaves.json) ---------------
    def union_layers(self, count):
        """The real generated attention through the real operation, one layer at a time."""
        names = [f'qsa.{i}' for i in range(count)]
        mappings = {name: (3 + i, 7 + i, 9 + i) for i, name in enumerate(names)}
        operation, lane, transaction, sources = self.prepare(mappings)

        def run(index):
            name = names[index]
            self.arena.storage.clear()
            self.open_layer(operation, lane, transaction, name, sources[name], mappings[name])
            logical = T((2, 4), device='cuda:0', dtype='torch.int32',
                        values=[index * 4, 4090 + index, 50, -1, 900 + index, 8100, 51, -1])
            query = T((2, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
            output = T((2, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
            rows = T((2,), device='cuda:0', dtype='torch.int64', values=[0, 1])
            self.run_generator(operation, self.attention(query, sources[name], logical, self.table, rows, 2,
                                                         self.arena, True, output, e3op=operation))
            operation.end_layer(name)
            return operation._layers[name]
        return operation, run

    def hold_incomplete(self, run):
        """Keep the layer's last consumer event incomplete: out of ready_all's reach."""
        event = run.consumed[-1].event
        event.ready = False
        self.device.events.remove(event)
        return event

    def test_union_sets_retire_once_their_last_consumer_is_complete(self):
        operation, run = self.union_layers(4)
        layers, single = [], None
        for index in range(4):
            layers.append(run(index))
            single = single or operation.stats['union_scratch_bytes']
            self.assertEqual(operation.stats['union_scratch_sets'], index + 1)
            self.assertEqual(operation.stats['union_scratch_retired'], index)
            self.assertEqual(operation.stats['union_scratch_live_bytes'], single)
            self.assertEqual([r.sparse.union is None for r in layers], [True] * index + [False])
        # The largest single set keeps its meaning; the aggregate never held two sets.
        self.assertEqual(operation.stats['union_scratch_bytes'], single)
        self.assertEqual(operation.stats['union_scratch_live_max_bytes'], single)
        inventory = operation.inventory()
        self.assertEqual((inventory['union_scratch_sets'], inventory['union_scratch_retired'],
                          inventory['union_scratch_live_max_bytes']), (4, 3, single))
        operation.end_forward()
        self.device.ready_all()
        self.assertTrue(operation.drain())
        operation.release()
        self.assertIsNone(layers[-1].sparse.union)
        self.assertEqual(operation._union_sets, [])

    def test_union_set_stays_while_its_last_consumer_is_incomplete(self):
        operation, run = self.union_layers(3)
        first = run(0)
        single = operation.stats['union_scratch_bytes']
        kept = first.sparse.union
        event = self.hold_incomplete(first)
        queries = self.device.queries
        second = run(1)
        # Never waited for: one nonblocking query, the set stays, the aggregate shows both.
        self.assertGreater(self.device.queries, queries)
        self.assertEqual(self.device.waits.count(event), 0)
        self.assertIs(first.sparse.union, kept)
        self.assertEqual(operation.stats['union_scratch_retired'], 0)
        self.assertEqual(operation.stats['union_scratch_live_max_bytes'], 2 * single)
        event.ready = True
        third = run(2)
        self.assertIsNone(first.sparse.union)
        self.assertIsNone(second.sparse.union)
        self.assertIsNotNone(third.sparse.union)
        self.assertEqual(operation.stats['union_scratch_retired'], 2)
        self.assertEqual(operation.stats['union_scratch_live_bytes'], single)
        self.assertEqual(operation.stats['union_scratch_live_max_bytes'], 2 * single)

    def test_union_set_stays_before_its_readback_and_under_row1_qualification(self):
        operation, run = self.union_layers(2)
        first = run(0)
        event = self.hold_incomplete(first)
        run(1)
        event.ready = True
        kept = first.sparse.union
        # The row1 oracle reads the layer at release: every set stays until then.
        operation.qualification = object()
        operation._retire_union_scratch()
        self.assertIs(first.sparse.union, kept)
        operation.qualification = None
        # A set whose selection readback was never host-observed stays too.
        count, first.sparse.selection_count = first.sparse.selection_count, None
        operation._retire_union_scratch()
        self.assertIs(first.sparse.union, kept)
        first.sparse.selection_count = count
        operation._retire_union_scratch()
        self.assertIsNone(first.sparse.union)
        self.assertEqual(operation.stats['union_scratch_retired'], 1)

    def test_alloc_receipt_is_one_line_per_release_behind_the_receipt_gate(self):
        for gate, lines in ((True, 1), (False, 0)):
            with self.subTest(gate=gate):
                self.device.native_memory_receipts = gate
                operation, run = self.union_layers(2)
                run(0)
                run(1)
                operation.note_indexer_scores(1024, 100000, 8, 4096, 64, 512)
                operation.end_forward()
                self.device.ready_all()
                self.assertTrue(operation.drain())
                with self.assertLogs('qsa_operation', level='DEBUG') as logs:
                    logging.getLogger('qsa_operation').debug('anchor')
                    operation.release()
                rows = [line for line in logs.output if 'flash-s5 native-alloc ' in line]
                self.assertEqual(len(rows), lines)
                self.assertEqual(operation.stats['alloc_receipt_errors'], 0)
                if lines:
                    single = operation.stats['union_scratch_bytes']
                    self.assertIn(f"union_sets=2 union_retired=1 union_single={single} union_live_max={single} ", rows[0])
                    self.assertIn('indexer_layers=1 indexer_rows=1024 indexer_width=12544 ', rows[0])
                    self.assertLess(len(rows[0]), 400)

    def test_descriptor_cap_uses_bounded_waves_before_one_attention(self):
        mapping = (5, 7)
        tokens = len(mapping) * PAGE_TOKENS
        # Fragmented rows overflow the strict plan but retain one attention bucket.
        selected = [2 * j for j in range(1250)] + [PAGE_TOKENS + 2 * j for j in range(625)]
        width = 1876
        operation, lane, transaction, sources = self.prepare({'qsa.0': mapping})
        hold = self.open_layer(operation, lane, transaction, 'qsa.0',
                               sources['qsa.0'], mapping)
        logical = T((1, width), device='cuda:0', dtype='torch.int32',
                    values=selected + [-1])
        query = T((1, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
        output = T((1, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
        rows = T((1,), device='cuda:0', dtype='torch.int64', values=[0])
        generator = self.attention(query, sources['qsa.0'], logical, self.table,
                                   rows, 1, self.arena, True, output,
                                   e3op=operation)
        operation.attach_continuation(generator)
        waves = []
        while True:
            before = len(self.device.copies)
            try:
                value = operation.advance()
            except StopIteration:
                break
            if getattr(value, 'qsa_schedule_boundary', False):
                self.device.ready_all()
                self.authorize_unit_profile(operation)
                continue
            waves.append(len(self.device.copies) - before)
            self.assertLessEqual(waves[-1], shd.MAX_DESCRIPTORS)
            if getattr(value, 'sparse_copy_pending', False):
                self.assertEqual(len(self.kernel.calls), 0)
                self.assertEqual(hold.stats['published'], 0)
            self.device.ready_all()
        self.assertEqual(waves, [256] * 7 + [83])
        self.assertEqual(len(self.kernel.calls), 1)
        self.assertEqual(hold.stats['published'], 1)
        self.assertEqual(operation.stats['sparse_descriptor_high_water'], 256)
        self.assertEqual(operation.stats['sparse_fallback'], 0)
        self.assertEqual(self.arena_rows(len(selected)),
                         [struct.pack('<Q', row) * 4
                          for row in self.expected_rows(selected, mapping)])
        self.assertEqual(len(operation._records), 4)
        operation.end_layer('qsa.0')
        operation.end_forward()
        self.assertTrue(operation.drain())
        operation.release()

    def test_cancel_mid_fill_drains_then_releases_and_refuses_reuse(self):
        pages = 124
        mapping = tuple(range(pages))
        tokens = pages * PAGE_TOKENS
        nrows, width = 60, 2052
        rows_values = [[r * (width - 1) + c for c in range(width - 1)]
                       for r in range(nrows)]
        operation, lane, transaction, sources = self.prepare({'qsa.0': mapping})
        source = sources['qsa.0']
        hold = self.open_layer(operation, lane, transaction, 'qsa.0', source,
                               mapping)
        flat = []
        for row in rows_values:
            flat.extend(row)
            flat.append(-1)
        logical = T((nrows, width), device='cuda:0', dtype='torch.int32',
                    values=flat)
        query = T((nrows, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
        output = T((nrows, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
        rows = T((nrows,), device='cuda:0', dtype='torch.int64',
                 values=list(range(nrows)))
        generator = self.attention(query, source, logical, self.table, rows,
                                   pages - 1, self.arena, True, output,
                                   e3op=operation)
        operation.attach_continuation(generator)
        self.assertTrue(operation.advance().qsa_schedule_boundary)
        self.authorize_unit_profile(operation)
        operation.advance()               # bucket 0 filled and submitted
        self.device.ready_all()
        operation.advance()               # bucket 1 filled and submitted
        operation.cancel('operator')
        with self.assertRaises(qo.OperationCancelled):
            operation.advance()
        self.device.ready_all()
        self.assertTrue(operation.drain())
        self.assertEqual(operation.stats['sparse_buckets'], 2)
        operation.release()
        self.assertTrue(hold.released)
        with self.assertRaises(shd.SourceRefused):
            hold.begin_bucket(9, 1, 1, shd.ROW_BYTES)
        with self.assertRaises(qo.OperationError):
            operation.begin_layer('qsa.0', source)

    def test_stale_allocator_lease_quarantines_before_fill(self):
        class LostLease(FakeAllocator):
            def assert_retained(self, reference):
                raise shd.SourceRefused('allocator lease lost')

        mapping = (3, 7, 9)
        operation, lane, transaction, sources = self.prepare({'qsa.0': mapping})
        source = sources['qsa.0']
        operation.begin_layer('qsa.0', source)
        operation.producer_submitted('qsa.0', source)
        operation.bind_plan('qsa.0', self.table, len(mapping),
                            len(mapping) * PAGE_TOKENS, self.arena)
        hold = shd.SparseSourceHold(
            make_owner(lane, 'qsa.0', mapping, len(mapping) * PAGE_TOKENS,
                       transaction=transaction),
            allocator=LostLease())
        operation.attach_sparse_source('qsa.0', hold)
        operation.bind_sparse_context('qsa.0', self.arena, self.table, min_pages=1)
        logical = T((2, 2), device='cuda:0', dtype='torch.int32',
                    values=[0, -1, 4090, -1])
        query = T((2, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
        output = T((2, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
        rows = T((2,), device='cuda:0', dtype='torch.int64', values=[0, 1])
        generator = self.attention(query, source, logical, self.table, rows, 2,
                                   self.arena, True, output, e3op=operation)
        operation.attach_continuation(generator)
        self.assertTrue(operation.advance().qsa_schedule_boundary)
        self.authorize_unit_profile(operation)
        with self.assertRaises(shd.SourceRefused):
            operation.advance()
        self.assertTrue(operation._failed)
        self.assertEqual(self.device.copies, [])
        with self.assertRaises(et.E3Pending):
            operation.release()

    def test_helper_binds_dense_plan_then_sparse_context(self):
        # 97,169 query rows is just past 24 pages; the helper itself binds the
        # dense plan and then the compact-row context before the pinned call.
        tokens = 24 * PAGE_TOKENS + 17
        pages = (tokens + PAGE_TOKENS - 1) // PAGE_TOKENS
        mapping = tuple(range(pages))
        operation, lane, transaction, sources = self.prepare({'qsa.0': mapping})
        source = sources['qsa.0']
        self.device.auto_ready = False
        operation.begin_layer('qsa.0', source)
        operation.producer_submitted('qsa.0', source)
        hold = self.hold_for(lane, transaction, 'qsa.0', mapping,
                             tokens)
        operation.attach_sparse_source('qsa.0', hold)
        # The helper itself binds the dense plan and the compact-row context.
        layer = types.SimpleNamespace(
            layer_name='qsa.0', _qsa_kv_offload=True, _qsa_stage_min_rows=1,
            _qsa_stage_arena_bytes=shd.ARENA_ROWS * shd.ROW_BYTES)
        width = 2
        flat = []
        for row in range(tokens):
            flat.append(row % 60000)
            flat.append(-1)
        logical = T((tokens, width), device='cuda:0', dtype='torch.int32',
                    values=flat)
        query = T((tokens, QUERY_HEADS, HEAD_DIM), device='cuda:0',
                  dtype='torch.bfloat16')
        output = T((tokens, QUERY_HEADS, HEAD_DIM), device='cuda:0',
                   dtype='torch.bfloat16')
        metadata = types.SimpleNamespace(
            _qsa_host_meta=((0, tokens), (tokens,)),
            query_start_loc=T((2,), device='cpu', dtype='torch.int32'),
            seq_lens=T((1,), device='cpu', dtype='torch.int32'),
            block_table=T((1, 260), device='cuda:0', dtype='torch.int64',
                          values=[0] * 260))
        generator = self.helper(operation, layer, query, source, logical,
                                metadata, tokens, True, output)
        operation.attach_continuation(generator)
        sentinels = 0
        while True:
            try:
                value = operation.advance()
            except StopIteration:
                break
            if getattr(value, 'qsa_schedule_boundary', False):
                sentinels += 1
                self.device.ready_all()
                self.authorize_unit_profile(operation)
                continue
            self.device.ready_all()
        self.assertEqual(sentinels, 1)
        self.assertEqual(operation.stats['sparse_buckets'], 1)
        run = operation._layers['qsa.0']
        self.assertEqual(run.pages, mapping)
        self.assertEqual(run.valid_tokens, tokens)
        plan = run.sparse.plan
        self.assertEqual(plan.bucket_rows, (60000,))
        self.assertEqual(self.arena_digest(len(plan.selected)),
                         self.expected_digest(plan.selected, mapping))
        operation.end_layer('qsa.0')
        operation.end_forward()
        self.device.ready_all()
        self.assertTrue(operation.drain())
        operation.release()

    def test_all_padding_selection_takes_the_empty_exit_without_dma(self):
        mapping = (3, 7, 9)
        operation, lane, transaction, sources = self.prepare({'qsa.0': mapping})
        source = sources['qsa.0']
        self.open_layer(operation, lane, transaction, 'qsa.0', source, mapping)
        logical = T((2, 3), device='cuda:0', dtype='torch.int32',
                    values=[-1, -1, -1, -1, -1, -1])
        query = T((2, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
        output = T((2, QUERY_HEADS, HEAD_DIM), device='cuda:0', dtype='torch.bfloat16',
                   values=[7.0] * (2 * QUERY_HEADS * HEAD_DIM))
        rows = T((2,), device='cuda:0', dtype='torch.int64', values=[0, 1])
        generator = self.attention(query, source, logical, self.table, rows, 2,
                                   self.arena, True, output, e3op=operation)
        operation.attach_continuation(generator)
        self.assertTrue(operation.advance().qsa_schedule_boundary)
        self.authorize_unit_profile(operation)
        with self.assertRaises(StopIteration):
            operation.advance()
        self.assertEqual(operation.stats['sparse_empty'], 1)
        self.assertEqual(self.device.copies, [])
        self.assertEqual(output.values, [0] * (2 * QUERY_HEADS * HEAD_DIM))
        operation.end_layer('qsa.0')
        operation.end_forward()
        operation.release()


class _PoolBlock:
    """The allocator's real block object: identity plus its live refcount."""

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 1
        self.is_null = False


class _BlockPool:
    """Engine allocator surface B's SourceHold proves against: ``blocks[id]``."""

    def __init__(self, ids):
        self.blocks = [None] * (max(ids) + 1)
        for block_id in ids:
            self.blocks[block_id] = _PoolBlock(block_id)
        self.freed = []

    def free_blocks(self, blocks):
        self.freed.extend(blocks)


class _ComposedAllocator:
    """Local worker allocator: registration identities plus retained references.

    It is deliberately not the engine pool: the worker never receives the
    engine's block objects or its live ``SchedulerLeases`` instance.
    """

    def __init__(self, versions, allocation_incarnation=11):
        self.versions = tuple(versions)
        self.allocation_incarnation = allocation_incarnation
        self.registrations = []
        self.refs = {}
        self.released = []

    def register_source(self, source, pages):
        self.registrations.append((source, tuple(pages)))
        return shd.AllocationRegistration(self.allocation_incarnation,
                                          self.versions)

    def retain(self, owner):
        reference = object()
        self.refs[id(reference)] = owner
        return reference

    def assert_retained(self, reference):
        if id(reference) not in self.refs:
            raise RuntimeError('local allocator lease is not retained')

    def release(self, reference):
        self.released.append(reference)
        self.refs.pop(id(reference), None)


class ComposedSourceLeaseTests(_GeneratedHarness, unittest.TestCase):
    """B's real scheduler hold + C's real worker adapter and operation.

    The engine half is ``scheduler_hooks.SchedulerLeases``/``SourceHold`` over
    real block objects; the worker half sees only the serialisable
    ``wire_v1.SourceLeaseDescriptor`` and its own ``LocalSourceRegistry``.  No
    test calls an engine Python object from the worker side, and local release
    never frees the engine's scheduler hold: only the all-rank barrier does.
    """

    def world(self):
        transaction = et.TransactionId(
            engine_epoch=1, request_uid='req-lease', request_generation=0,
            chunk_start=0, chunk_end=4, txn_seq=7)
        lane = make_lane()
        pages = (3, 7, 9)
        state_epoch = 2
        admission = wire_v1.AdmissionRecord(
            transaction=transaction, prompt_end=6, state_epoch=state_epoch,
            committed_end=0, block_tables=(pages,),
            slot_mappings=((0, 1, 2, 3),), new_block_ids_to_zero=(),
            cow_pairs=(), input_ids=(10, 11, 12, 13))
        pool = _BlockPool(pages)
        objects = tuple(pool.blocks[page] for page in pages)
        leases = scheduler_hooks.SchedulerLeases()
        leases.attach(types.SimpleNamespace(), pool)
        leases.reserve_blocks(transaction, LEASE_CATEGORY.SOURCE_HISTORY,
                              pages, blocks=objects)
        ticket = et.OperationTicket(transaction, 0, lane, 0,
                                    et.DependencyRole.PRODUCER)
        leases.bind(ticket, LEASE_CATEGORY.SOURCE_HISTORY)
        engine_hold = scheduler_hooks.SourceHold(
            leases, transaction=transaction, admission=admission, pool=pool,
            blocks=objects, ranks=(0,))
        engine_hold.bind_ticket(ticket)
        engine_hold.bind_owners({0: lane}, 1)
        descriptor = engine_hold.descriptor()
        chunk = worker_bridge.admission_to_chunk(admission)
        registry = shd.LocalSourceRegistry(lane)
        source = self.make_source(max(pages) + 1)
        allocator = _ComposedAllocator(range(1, 16))
        registration = registry.register(
            layer='qsa.0', group=0, transaction=transaction,
            state_epoch=state_epoch, valid_tokens=len(pages) * PAGE_TOKENS,
            physical_pages=pages, source=source, allocator=allocator)
        opened = registry.open(descriptor, chunk, layer='qsa.0', group=0,
                               allocator=allocator,
                               registration=registration)
        return types.SimpleNamespace(
            transaction=transaction, lane=lane, pages=pages, pool=pool,
            objects=objects, leases=leases, ticket=ticket,
            engine_hold=engine_hold, descriptor=descriptor, admission=admission,
            chunk=chunk, registry=registry, source=source, allocator=allocator,
            registration=registration, opened=opened)

    def operation_for(self, world):
        operation = qo.QsaPrefillOperation(
            world.transaction, world.lane, device=self.device,
            plan_provider=lambda name, count: world.pages[:count],
            source_epochs={'qsa.0': world.registration.source_epoch})
        operation.state_epoch = world.admission.state_epoch
        operation.register_source(
            'qsa.0', world.source,
            source_epoch=world.registration.source_epoch,
            versions=world.registration.versions)
        operation.begin_forward(layers=('qsa.0',))
        operation.begin_layer('qsa.0')
        operation.producer_submitted('qsa.0', world.source)
        operation.bind_plan('qsa.0', self.table, len(world.pages),
                            len(world.pages) * PAGE_TOKENS, self.arena)
        operation.attach_sparse_source('qsa.0', world.opened.hold)
        operation.bind_sparse_context('qsa.0', self.arena, self.table,
                                      min_pages=1)
        return operation

    def attention_run(self, world, operation, *, output=None):
        logical = T((2, 4), device='cuda:0', dtype='torch.int32',
                    values=[0, 4047, 4048, -1, 8095, 4048, 4047, -1])
        query = T((2, QUERY_HEADS, HEAD_DIM), device='cuda:0',
                  dtype='torch.bfloat16')
        output = output or T((2, QUERY_HEADS, HEAD_DIM), device='cuda:0',
                             dtype='torch.bfloat16')
        rows = T((2,), device='cuda:0', dtype='torch.int64', values=[0, 1])
        return self.attention(query, world.source, logical, self.table, rows,
                              len(world.pages) - 1, self.arena, True, output,
                              e3op=operation)

    def engine_barrier(self, world):
        proof = et.CompletionProof(world.ticket, et.CompletionKind.COPY_EVENT, 3)
        acks = {0: wire_v1.ReleaseAck(world.transaction, 0, 2, 3,
                                      released=True, request_closed=False)}
        return world.engine_hold.release(proof, acks, (0,))

    def test_worker_adapter_and_engine_hold_join_without_a_borrowed_pin(self):
        world = self.world()
        # Identities come from the allocator's registration event, not from a
        # constant relabelled as an incarnation.
        self.assertEqual(world.registration.allocation_incarnation, 11)
        self.assertEqual(world.registration.versions, tuple(range(1, 16)))
        self.assertEqual(world.allocator.registrations,
                         [(world.source, world.pages)])
        self.assertEqual(world.opened.versions, world.registration.versions)
        self.assertEqual(world.opened.source_epoch,
                         world.registration.source_epoch)
        self.assertEqual(world.descriptor.block_ids, world.pages)
        self.assertEqual(world.descriptor.consumer_ranks, (0,))

        operation = self.operation_for(world)
        selections, buckets = self.run_generator(
            operation, self.attention_run(world, operation))
        self.assertEqual((selections, buckets), (0, 1))
        self.assertEqual(operation.stats['sparse_routes'], 1)
        self.assertEqual(self.device.page_copies, [])
        run = operation._layers['qsa.0']
        self.assertEqual(run.sparse.plan.selected, (0, 4047, 4048, 8095))
        expected = [3 * PAGE_TOKENS, 3 * PAGE_TOKENS + 4047,
                    7 * PAGE_TOKENS, 7 * PAGE_TOKENS + 4047]
        self.assertEqual([copy[0] for copy in self.device.copies], expected)
        self.assertEqual(self.arena.storage.rows[:len(expected)],
                         [struct.pack('<Q', row) * 4 for row in expected])
        self.assertEqual(run.sparse.hold.stats['consumed'], 1)
        self.assertEqual(run.sparse.hold.stats['abandoned'], 0)

        operation.end_layer('qsa.0')
        operation.end_forward()
        self.device.ready_all()
        self.assertTrue(operation.drain())
        operation.release()
        # The local completion query released only the worker's view.
        self.assertTrue(world.opened.hold.released)
        self.assertEqual(len(world.allocator.released), 1)
        world.registry.close(world.opened)
        self.assertEqual(world.leases.held_blocks(), world.pages)
        self.assertEqual(world.leases.held_transaction(
            world.transaction, LEASE_CATEGORY.SOURCE_HISTORY), world.pages)
        self.assertTrue(world.engine_hold.validate(world.lane, 0, world.pages))
        with self.assertRaises(shd.SourceRefused):
            world.opened.hold.validate(world.opened.hold.owner, 'qsa.0',
                                       world.pages)

        # Before the real barrier the engine allocator may not reuse a page.
        self.assertEqual(world.leases.hold_leased_blocks(world.objects), ())
        self.assertEqual(set(world.leases.pending_free), set(world.pages))
        with self.assertRaises(et.E3ContractError):
            world.leases.note_allocated(world.objects)

        # The validated all-rank worker ACK/barrier is what frees the source.
        self.assertTrue(self.engine_barrier(world))
        self.assertEqual(world.leases.held_blocks(), ())
        self.assertEqual([block.block_id for block in world.pool.freed],
                         list(world.pages))
        world.leases.note_allocated(world.objects)
        self.assertEqual(world.leases.hold_leased_blocks(world.objects),
                         world.objects)
        with self.assertRaises(et.E3ContractError):
            world.engine_hold.validate(world.lane, 0, world.pages)
        with self.assertRaises(et.E3ContractError):
            world.engine_hold.descriptor()

    def test_source_lease_fields_are_the_transport_contract(self):
        names = tuple(field.name for field in
                      dataclasses.fields(wire_v1.SourceLeaseDescriptor))
        self.assertEqual(names, shd.SOURCE_LEASE_FIELDS)
        world = self.world()
        self.assertEqual(
            (world.descriptor.transaction, world.descriptor.state_epoch,
             world.descriptor.block_ids, world.descriptor.consumer_ranks),
            (world.transaction, 2, world.pages, (0,)))
        self.assertTrue(world.descriptor.validate_frame(world.admission, (0,)))
        self.assertTrue(world.descriptor.validate_frame(world.chunk, (0,)))

    def test_open_refuses_every_wrong_frame_before_any_dma(self):
        world = self.world()
        foreign = et.TransactionId(engine_epoch=1, request_uid='req-other',
                                   request_generation=0, chunk_start=0,
                                   chunk_end=4, txn_seq=8)
        cases = (
            (None, world.chunk, 'SourceLeaseDescriptor is required'),
            (object(), world.chunk, 'SourceLeaseDescriptor is required'),
            (world.descriptor, None, 'admitted frame is required'),
            (wire_v1.SourceLeaseDescriptor(foreign, 2, world.pages, (0,)),
             world.chunk, 'transaction differs'),
            (wire_v1.SourceLeaseDescriptor(world.transaction, 3, world.pages,
                                           (0,)),
             world.chunk, 'state epoch differs'),
            (wire_v1.SourceLeaseDescriptor(world.transaction, 2, world.pages,
                                           (1,)),
             world.chunk, 'does not name the local rank'),
            (wire_v1.SourceLeaseDescriptor(world.transaction, 2, (3, 7), (0,)),
             world.chunk, 'omits a local source page'),
            (wire_v1.SourceLeaseDescriptor(world.transaction, 2,
                                           (3, 7, 9, 11), (0,)),
             world.chunk, 'frame differs from the admitted chunk'),
        )
        for descriptor, admission, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(shd.SourceRefused, message):
                    world.registry.open(descriptor, admission, layer='qsa.0',
                                        group=0, allocator=world.allocator)
        self.assertEqual(self.device.copies, [])
        self.assertEqual(world.allocator.registrations,
                         [(world.source, world.pages)])
        self.assertTrue(world.opened.hold.releasable)
        # A changed admitted row for the same registration is a wrong map.
        other = wire_v1.AdmissionRecord(
            transaction=world.transaction, prompt_end=6, state_epoch=2,
            committed_end=0, block_tables=((3, 7, 8),),
            slot_mappings=((0, 1, 2, 3),), new_block_ids_to_zero=(),
            cow_pairs=(), input_ids=(10, 11, 12, 13))
        with self.assertRaisesRegex(shd.SourceRefused,
                                    'registered map differs'):
            world.registry.open(world.descriptor,
                                worker_bridge.admission_to_chunk(other),
                                layer='qsa.0', group=0,
                                allocator=world.allocator)
        # A group the admitted frame does not have is refused before DMA.
        world.registry.register(
            layer='qsa.0', group=1, transaction=world.transaction,
            state_epoch=2, valid_tokens=len(world.pages) * PAGE_TOKENS,
            physical_pages=world.pages, source=world.source,
            allocator=world.allocator)
        with self.assertRaisesRegex(shd.SourceRefused, 'outside the admitted frame'):
            world.registry.open(world.descriptor, world.chunk, layer='qsa.0',
                                group=1, allocator=world.allocator)

    def test_rebound_and_stale_registrations_are_refused(self):
        world = self.world()
        with self.assertRaisesRegex(shd.SourceRefused, 'already open'):
            world.registry.open(world.descriptor, world.chunk, layer='qsa.0',
                                group=0, allocator=world.allocator)
        with self.assertRaisesRegex(shd.SourceRefused, 'still retained'):
            world.registry.close(world.opened)
        with self.assertRaisesRegex(shd.SourceRefused, 'rebound'):
            world.registry.register(
                layer='qsa.0', group=0, transaction=world.transaction,
                state_epoch=2, valid_tokens=len(world.pages) * PAGE_TOKENS,
                physical_pages=world.pages, source=self.make_source(10),
                allocator=world.allocator)
        world.opened.hold.release()
        world.registry.close(world.opened)
        newer_source = self.make_source(10)
        newer = world.registry.register(
            layer='qsa.0', group=0, transaction=world.transaction,
            state_epoch=2, valid_tokens=len(world.pages) * PAGE_TOKENS,
            physical_pages=world.pages, source=newer_source,
            allocator=world.allocator)
        self.assertNotEqual(newer.source_epoch, world.registration.source_epoch)
        self.assertNotEqual(newer.source_incarnation,
                            world.registration.source_incarnation)
        with self.assertRaisesRegex(shd.SourceRefused, 'stale local source'):
            world.registry.open(world.descriptor, world.chunk, layer='qsa.0',
                                group=0, allocator=world.allocator,
                                registration=world.registration)
        reopened = world.registry.open(world.descriptor, world.chunk,
                                       layer='qsa.0', group=0,
                                       allocator=world.allocator,
                                       registration=newer)
        self.assertEqual(reopened.source_epoch, newer.source_epoch)
        self.assertEqual(reopened.versions, newer.versions)
        reopened.hold.release()
        world.registry.close(reopened)

    def test_cancel_before_first_bucket_releases_only_the_local_view(self):
        world = self.world()
        self.device.auto_ready = False
        operation = self.operation_for(world)
        generator = self.attention_run(world, operation)
        operation.attach_continuation(generator)
        value = operation.advance()
        self.assertTrue(getattr(value, 'qsa_schedule_boundary', False))
        self.assertEqual(operation.pending_selection, 1)
        self.assertEqual(world.opened.hold.stats['buckets'], 0)
        self.assertEqual(self.device.copies, [])
        operation.cancel('operator cancel')
        self.device.ready_all()
        self.assertTrue(operation.drain())
        operation.release()
        self.assertTrue(world.opened.hold.released)
        self.assertEqual(world.opened.hold.stats['buckets'], 0)
        self.assertEqual(world.opened.hold.stats['abandoned'], 0)
        world.registry.close(world.opened)
        # Cancellation does not free the engine's scheduler hold; the engine
        # barrier (here from the same process) remains the only release.
        self.assertEqual(world.leases.held_blocks(), world.pages)
        self.assertTrue(self.engine_barrier(world))
        self.assertEqual(world.leases.held_blocks(), ())


class PackageRegistrationTests(unittest.TestCase):
    def test_sparse_module_is_registered_in_the_package_inventory(self):
        import decode_probe
        import make_s5_overlap
        self.assertIn('sparse_history_dma', decode_probe.GENERATED_SOURCES)
        self.assertIn('sparse_history_dma', make_s5_overlap.E3_MODULES)
        self.assertNotIn('sparse_history_dma', decode_probe.BUILD_ONLY_SOURCES)


if __name__ == '__main__':
    unittest.main()
