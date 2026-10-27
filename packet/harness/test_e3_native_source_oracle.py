"""Native source lease construction and the real row1 byte oracle.

CPU-only acceptance for the worker half of the E3 source lease and for the
serial native row1 qualification:

* ``verify_kv_layout`` proves the admitted QSA mapping from the pinned
  ``BlockTables`` (factor 1 is verified, never assumed; an expanded mapping is
  refused);
* ``bind_local_sources`` turns one engine ``SourceLeaseDescriptor`` and the
  worker's actual pinned ``_qsa_host_kv`` tensors into per-layer
  ``SparseSourceHold`` objects, with rollback on a partial open and the exact
  epoch/version census;
* the engine's *first PREPARE* command carries the scheduler hold's actual
  descriptor, and the real ``NativeWorkerBridge`` binds it over the real
  ``QsaLayerDriver`` before any segment is submitted;
* the row1 oracle compares the actual arena bytes against the original UVA
  gather in the same arena and the QSA output before MoE against the original
  ``flash_e2_arena_attention``, through the E2 exactness snapshot mechanism.

Every tensor, event, stream, transport and device here is a fake. No GPU, CUDA,
vLLM process or PP traffic is exercised; nothing in this file is native
execution evidence, and the shipping performance path never runs the oracle.
"""
import hashlib
import math
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import engine_driver as ed  # noqa: E402
import e3_types as et  # noqa: E402
import lane_dispatch  # noqa: E402
import native_resources  # noqa: E402
import native_source as ns  # noqa: E402
import operation_source  # noqa: E402
import qsa_driver  # noqa: E402
import qsa_operation as qo  # noqa: E402
import row1_oracle as ro  # noqa: E402
import sparse_history_dma as shd  # noqa: E402
import wire_v1 as w  # noqa: E402
import worker_bridge  # noqa: E402

# The reviewed fixtures and fakes (fakes only; no device is touched).
import test_e3_admission_native as admission_test  # noqa: E402
import test_e3_r4_worker_bridge as r4  # noqa: E402
import test_e3_sparse_history_dma as sparse_test  # noqa: E402

T = sparse_test.T
Rows = sparse_test.Rows
PAGE_ROWS = shd.PAGE_ROWS
ROW_WIDTH = sparse_test.ROW_WIDTH
ROW_BYTES = shd.ROW_BYTES
ARENA_PAGES = shd.ARENA_PAGES
ARENA_ROWS = shd.ARENA_ROWS
QUERY_HEADS = sparse_test.QUERY_HEADS
HEAD_DIM = sparse_test.HEAD_DIM


# --- fixtures -----------------------------------------------------------------

class _SourceLayer:
    """One offloaded QSA layer module with its actual pinned history tensor."""

    def __init__(self, name, source):
        self.layer_name = name
        self._qsa_kv_offload = True
        self._qsa_host_kv = source


class SourceRunner:
    """The constructed-runner surface ``bind_local_sources`` proves against."""

    def __init__(self, layer_names=('qsa.0', 'qsa.1'), *, pool=8,
                 group_names=None, block_size=PAGE_ROWS, bpk=(1,),
                 block_sizes=(PAGE_ROWS,), kernel=(PAGE_ROWS,),
                 with_tables=True):
        self.layer_names = tuple(layer_names)
        self.sources = {
            name: T((pool, 1, PAGE_ROWS, ROW_WIDTH), device='cpu',
                    dtype='torch.bfloat16', pinned=True)
            for name in self.layer_names}
        self.layers = [_SourceLayer(name, self.sources[name])
                       for name in self.layer_names]
        self.model = types.SimpleNamespace(
            modules=lambda: iter(self.layers))
        names = tuple(group_names or self.layer_names)
        self.kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=(types.SimpleNamespace(
                layer_names=names,
                kv_cache_spec=types.SimpleNamespace(block_size=block_size)),))
        if with_tables:
            self.block_tables = types.SimpleNamespace(
                blocks_per_kv_block=tuple(bpk),
                block_sizes=tuple(block_sizes),
                kernel_block_sizes=tuple(kernel))


def make_admission(*, state_epoch=2, chunk_end=4, prompt_end=6,
                   block_tables=((3, 7, 9),), cow_pairs=(), transaction=None):
    transaction = transaction or sparse_test.make_transaction(
        chunk_end=chunk_end, engine_epoch=1)
    return w.AdmissionRecord(
        transaction=transaction, prompt_end=prompt_end, state_epoch=state_epoch,
        committed_end=transaction.chunk_start, block_tables=block_tables,
        slot_mappings=((0,),), new_block_ids_to_zero=(), cow_pairs=cow_pairs,
        input_ids=tuple(range(transaction.chunk_end - transaction.chunk_start)))


def make_descriptor(admission, *, block_ids=None, ranks=(0,),
                    state_epoch=None):
    return w.SourceLeaseDescriptor(
        admission.transaction,
        admission.state_epoch if state_epoch is None else state_epoch,
        tuple(sorted(block_ids if block_ids is not None
                     else {page for row in admission.block_tables for page in row
                           if page > 0})) or (1,),
        tuple(ranks))


def chunk_for(record):
    """The worker's frozen chunk for a wire admission (the binder's input)."""
    return worker_bridge.admission_to_chunk(record)


class _Dispatcher:
    """The minimal worker-local dispatcher surface the binder consumes."""

    def __init__(self, lane, runner):
        self.operation = types.SimpleNamespace(lane=lane)
        self.runner = runner
        self.source_registry = None
        self.source_allocator = None


# --- layout and allocator -----------------------------------------------------

class KvLayoutTests(unittest.TestCase):
    """The admitted mapping is proved from the pinned block tables."""

    def runner(self, **overrides):
        return SourceRunner(**overrides)

    def test_factor_one_layout_is_verified_for_every_qsa_layer(self):
        layout = ns.verify_kv_layout(
            self.runner(), ('qsa.0', 'qsa.1'))
        self.assertEqual(layout.group('qsa.0'), 0)
        self.assertEqual(layout.group('qsa.1'), 0)
        self.assertEqual(layout.blocks_per_kv_block, (1,))
        self.assertEqual(layout.block_sizes, (PAGE_ROWS,))
        self.assertEqual(layout.page_rows, PAGE_ROWS)

    def test_pinned_kv_config_uses_a_list_of_groups(self):
        runner = self.runner()
        runner.kv_cache_config.kv_cache_groups = list(runner.kv_cache_config.kv_cache_groups)
        self.assertEqual(ns.verify_kv_layout(runner, ('qsa.0',)).group('qsa.0'), 0)

    def test_unrelated_circular_buffer_geometry_is_not_a_qsa_page(self):
        runner = self.runner()
        ring = types.SimpleNamespace(layer_names=['indexer.ring'],
            kv_cache_spec=types.SimpleNamespace(block_size=8))
        runner.kv_cache_config.kv_cache_groups = (ring, *runner.kv_cache_config.kv_cache_groups)
        runner.block_tables = types.SimpleNamespace(blocks_per_kv_block=[1, 1],
            block_sizes=[8, PAGE_ROWS], kernel_block_sizes=[8, PAGE_ROWS])
        self.assertEqual(ns.verify_kv_layout(runner, ('qsa.0',)).group('qsa.0'), 1)

    def test_missing_block_table_layout_is_refused_not_assumed(self):
        runner = self.runner(with_tables=False)
        with self.assertRaises(ns.NativeLayoutError) as caught:
            ns.verify_kv_layout(runner, ('qsa.0',))
        self.assertIn('must be verified, not assumed', str(caught.exception))

    def test_expanded_blocks_per_kv_block_is_refused(self):
        # block_sizes // kernel_block_sizes == 2: a scheduler id would need an
        # expansion this 4048-row transport does not implement.
        runner = self.runner(bpk=(2,), block_sizes=(2 * PAGE_ROWS,),
                             kernel=(PAGE_ROWS,))
        with self.assertRaises(ns.NativeLayoutError) as caught:
            ns.verify_kv_layout(runner, ('qsa.0',))
        self.assertIn('expanded QSA block tables are unsupported',
                      str(caught.exception))

    def test_block_size_and_kernel_are_the_transport_page(self):
        with self.assertRaises(ns.NativeLayoutError) as caught:
            ns.verify_kv_layout(self.runner(block_size=2048), ('qsa.0',))
        self.assertIn('4048-row transport page', str(caught.exception))
        with self.assertRaises(ns.NativeLayoutError) as caught:
            ns.verify_kv_layout(self.runner(kernel=(2048,)), ('qsa.0',))
        self.assertIn('kernel block size', str(caught.exception))

    def test_inconsistent_block_sizes_are_refused(self):
        runner = self.runner(bpk=(1,), block_sizes=(2 * PAGE_ROWS,),
                             kernel=(PAGE_ROWS,))
        with self.assertRaises(ns.NativeLayoutError) as caught:
            ns.verify_kv_layout(runner, ('qsa.0',))
        self.assertIn('disagrees with the pinned block sizes',
                      str(caught.exception))
        two_groups = (types.SimpleNamespace(
            layer_names=('qsa.0',),
            kv_cache_spec=types.SimpleNamespace(block_size=PAGE_ROWS)),)
        runner = self.runner()
        runner.kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=two_groups * 2)
        with self.assertRaises(ns.NativeLayoutError) as caught:
            ns.verify_kv_layout(runner, ('qsa.0',))
        self.assertIn('disagrees with the KV groups', str(caught.exception))

    def test_a_layer_outside_the_admitted_groups_is_refused(self):
        with self.assertRaises(ns.NativeLayoutError) as caught:
            ns.verify_kv_layout(self.runner(), ('qsa.0', 'qsa.7'))
        self.assertIn('qsa.7', str(caught.exception))


class SourceGeometryTests(unittest.TestCase):
    """The real pinned tensor facts; no geometry-only or Boolean claim."""

    def source(self, **overrides):
        values = dict(device='cpu', dtype='torch.bfloat16', pinned=True)
        values.update(overrides)
        return T((8, 1, PAGE_ROWS, ROW_WIDTH), **values)

    def test_geometry_requires_a_pinned_contiguous_bf16_pool(self):
        self.assertEqual(ns.source_geometry(self.source()),
                         (8, 1, PAGE_ROWS, ROW_WIDTH))
        with self.assertRaises(ns.NativeSourceError) as caught:
            ns.source_geometry(self.source(pinned=False))
        self.assertIn('not pinned', str(caught.exception))
        with self.assertRaises(ns.NativeSourceError) as caught:
            ns.source_geometry(self.source(dtype='torch.float16'))
        self.assertIn('bf16', str(caught.exception))
        with self.assertRaises(ns.NativeSourceError) as caught:
            ns.source_geometry(self.source(strides=(1, 1, 1, 1)))
        self.assertIn('not contiguous', str(caught.exception))
        bad = T((8, 1, PAGE_ROWS, ROW_WIDTH - 1), device='cpu',
                dtype='torch.bfloat16', pinned=True)
        with self.assertRaises(ns.NativeSourceError) as caught:
            ns.source_geometry(bad)
        self.assertIn('invalid geometry', str(caught.exception))

    def test_pinned_flag_cannot_replace_the_cpu_device_proof(self):
        source = self.source()
        source.device = 'cuda:0'
        with self.assertRaisesRegex(ns.NativeSourceError, 'CPU storage'):
            ns.source_geometry(source)

    def test_a_geometry_only_object_is_not_a_pinned_proof(self):
        geometry = types.SimpleNamespace(
            shape=(8, 1, PAGE_ROWS, ROW_WIDTH), dtype='torch.bfloat16', device='cpu')
        with self.assertRaises(ns.NativeSourceError) as caught:
            ns.source_geometry(geometry)
        self.assertIn('pinned/contiguity proof', str(caught.exception))

    def test_storage_identity_is_the_allocation_not_the_geometry(self):
        first = self.source()
        second = self.source()
        self.assertGreater(ns.storage_identity(first), 0)
        self.assertNotEqual(ns.storage_identity(first),
                            ns.storage_identity(second))
        with self.assertRaises(ns.NativeSourceError) as caught:
            ns.storage_identity(T((1,), device='cpu', address=0))
        self.assertIn('allocation identity', str(caught.exception))


class NativeAllocatorTests(unittest.TestCase):
    """Retain/release over the actual pinned tensor, not scheduler block ids."""

    def setUp(self):
        self.source = T((12, 1, PAGE_ROWS, ROW_WIDTH), device='cpu',
                        dtype='torch.bfloat16', pinned=True)
        self.transaction = sparse_test.make_transaction(chunk_end=8096)
        self.lane = sparse_test.make_lane()
        self.allocator = ns.NativeHostSourceAllocator()

    def frame(self, *, transaction=None, pages=(3, 7)):
        return self.allocator.begin_frame(
            self.source, layer='qsa.0',
            transaction=transaction or self.transaction, state_epoch=4,
            pages=pages)

    def owner(self, frame, **overrides):
        values = dict(transaction=self.transaction, lane=self.lane,
                      layer='qsa.0', storage_epoch=self.lane.storage_epoch,
                      source_epoch=0, source_incarnation=1,
                      allocation_incarnation=frame.allocation_incarnation,
                      physical_pages=frame.pages, valid_tokens=8096,
                      mapping_version=1)
        values.update(overrides)
        return shd.SourceOwner(**values)

    def test_begin_frame_is_idempotent_and_versions_change_with_the_frame(self):
        frame = self.frame()
        self.assertEqual(frame.pages, (3, 7))
        versions = dict((page, frame.versions[page]) for page in frame.pages)
        same = self.frame()
        self.assertIs(same, frame)
        self.assertEqual(self.allocator.stats['registrations'], 1)
        self.assertEqual(self.allocator.stats['frames'], 1)
        later = self.frame(pages=(3, 8))
        self.assertEqual(later.allocation_incarnation,
                         frame.allocation_incarnation)
        self.assertGreater(later.versions[8], versions[3])
        self.assertNotEqual(later.versions[3], versions[3])
        self.assertEqual(self.allocator.stats['registrations'], 2)

    def test_a_different_tensor_is_a_new_allocation_incarnation(self):
        first = self.frame()
        other = T((12, 1, PAGE_ROWS, ROW_WIDTH), device='cpu',
                  dtype='torch.bfloat16', pinned=True)
        second = self.allocator.begin_frame(
            other, layer='qsa.0', transaction=self.transaction,
            state_epoch=4, pages=(2,))
        self.assertGreater(second.allocation_incarnation,
                           first.allocation_incarnation)
        self.assertEqual(second.versions[2], second.version)
        self.assertEqual(second.versions[3], 0,
                         'a fresh pool starts every page at version zero')

    def test_retain_proves_the_exact_live_object(self):
        frame = self.frame()
        registration = self.allocator.register_source(self.source, (3, 7))
        self.assertEqual(registration.allocation_incarnation,
                         frame.allocation_incarnation)
        self.assertEqual(registration.versions, frame.versions)
        reference = self.allocator.retain(self.owner(frame))
        self.assertTrue(self.allocator.assert_retained(reference))
        self.assertEqual(self.allocator.inventory()['live'], 1)
        self.allocator.release(reference)
        self.assertEqual(self.allocator.inventory()['live'], 0)
        with self.assertRaises(shd.SourceRefused) as caught:
            self.allocator.assert_retained(reference)
        self.assertIn('no longer retains', str(caught.exception))
        # A Boolean or unknown object is never a retained reference.
        with self.assertRaises(shd.SourceRefused):
            self.allocator.assert_retained(True)

    def test_retain_refuses_a_stale_or_foreign_owner(self):
        frame = self.frame()
        with self.assertRaises(shd.SourceRefused) as caught:
            self.allocator.retain(self.owner(frame, allocation_incarnation=99))
        self.assertIn('incarnation is stale', str(caught.exception))
        with self.assertRaises(shd.SourceRefused) as caught:
            self.allocator.retain(self.owner(frame, physical_pages=(3,)))
        self.assertIn('pages differ', str(caught.exception))
        foreign = shd.SourceOwner(
            replace(self.transaction, txn_seq=self.transaction.txn_seq + 1),
            self.lane, 'qsa.0',
            self.lane.storage_epoch, 0, 1, 1, (3, 7), 8096, 1)
        with self.assertRaises(shd.SourceRefused) as caught:
            self.allocator.retain(foreign)
        self.assertIn('no local registration', str(caught.exception))

    def test_released_chunk_metadata_does_not_accumulate_forever(self):
        for seq in range(16):
            transaction = replace(self.transaction, txn_seq=self.transaction.txn_seq + seq)
            frame = self.frame(transaction=transaction)
            owner = self.owner(frame, transaction=transaction)
            reference = self.allocator.retain(owner)
            self.allocator.release(reference)
        self.assertEqual(len(self.allocator._by_owner), 0)
        self.assertEqual(self.allocator.inventory()['live'], 0)
        self.assertEqual(len(self.allocator._frames), 1)

    def test_invalid_frame_pages_are_refused(self):
        for pages in ((3, 3), (99,), (3, 99)):
            with self.assertRaises(shd.SourceRefused):
                self.frame(pages=pages)


# --- worker binding -----------------------------------------------------------

class LocalSourceBindingTests(unittest.TestCase):
    """The real binder over the real pinned tensors and the admitted frame."""

    def setUp(self):
        self.runner = SourceRunner(pool=12)
        self.lane = sparse_test.make_lane()
        self.dispatcher = _Dispatcher(self.lane, self.runner)
        self.admission = make_admission()
        self.chunk = chunk_for(self.admission)
        self.descriptor = w.SourceLeaseDescriptor(
            self.admission.transaction, self.admission.state_epoch,
            (3, 7, 9), (0,))
        self.binding = ns.bind_local_sources(
            self.dispatcher, self.chunk, self.descriptor, rank=0)

    def tearDown(self):
        if self.dispatcher.source_registry is not None:
            for opened in self.binding.opened:
                if not opened.hold.released:
                    opened.hold.release()
            self.binding.close(self.dispatcher.source_registry)
        self.dispatcher.source_allocator = None
        self.dispatcher.source_registry = None

    def test_every_layer_holds_its_actual_tensor_and_the_admitted_pages(self):
        self.assertEqual(self.binding.layers, ('qsa.0', 'qsa.1'))
        self.assertEqual(self.binding.transaction, self.admission.transaction)
        self.assertEqual(self.binding.state_epoch, self.admission.state_epoch)
        self.assertEqual(self.binding.rank, 0)
        self.assertEqual(self.binding.layer_tables,
                         (('qsa.0', (3,)), ('qsa.1', (3,))))
        self.assertEqual(self.binding.group_rows,
                         (('qsa.0', 0, (3,)), ('qsa.1', 0, (3,))))
        for name in ('qsa.0', 'qsa.1'):
            hold = self.binding.holds[name]
            self.assertIs(type(hold), shd.SparseSourceHold)
            self.assertFalse(hold.released)
            self.assertEqual(hold.owner.physical_pages, (3,))
            self.assertEqual(hold.owner.valid_tokens, 4)
            self.assertIs(self.binding.sources[name], self.runner.sources[name])
            self.assertEqual(self.binding.source_epochs[name],
                             self.binding.registrations[name].source_epoch)
            versions = self.binding.versions[name]
            self.assertGreater(len(versions), 3)
            self.assertIs(self.dispatcher.source_registry.lane, self.lane)
            self.assertEqual(self.dispatcher.source_registry.blocks_per_kv_block, 1)
        self.assertEqual(self.dispatcher.source_allocator.inventory()['live'], 2)
        self.assertEqual(len(self.dispatcher.source_registry._active), 2)
        # Independent layers keep independent holds, owners and storage even
        # when the admitted physical page id happens to be the same.
        self.assertIsNot(self.binding.holds['qsa.0'],
                         self.binding.holds['qsa.1'])
        self.assertIsNot(self.binding.holds['qsa.0'].owner,
                         self.binding.holds['qsa.1'].owner)
        self.assertIsNot(self.binding.sources['qsa.0'],
                         self.binding.sources['qsa.1'])
        self.assertNotEqual(
            ns.storage_identity(self.binding.sources['qsa.0']),
            ns.storage_identity(self.binding.sources['qsa.1']))
        # The descriptor covers the whole admitted frame; the binder retains
        # exactly the row prefix the chunk's token frontier reaches.
        self.assertEqual(self.descriptor.block_ids, (3, 7, 9))
        self.assertEqual(self.binding.holds['qsa.0'].owner.physical_pages, (3,))
        self.assertEqual(self.binding.holds['qsa.1'].owner.physical_pages, (3,))
        # Every page is inside the descriptor and the pinned pool.
        for opened in self.binding.opened:
            self.assertEqual(opened.descriptor, self.descriptor)
            self.assertEqual(opened.local_rank, 0)

    def test_release_then_close_drops_every_local_view(self):
        registry = self.dispatcher.source_registry
        allocator = self.dispatcher.source_allocator
        for opened in self.binding.opened:
            opened.hold.release()
        self.binding.close(registry)
        self.assertTrue(self.binding.closed)
        self.assertEqual(registry._active, {})
        self.assertEqual(allocator.inventory()['live'], 0)
        self.assertEqual(allocator.inventory()['released'], 2)
        # Closing twice is inert, not a second release.
        self.assertTrue(self.binding.close(registry))

    def test_rebinding_while_the_local_lease_is_open_is_refused(self):
        with self.assertRaises(shd.SourceRefused) as caught:
            ns.bind_local_sources(self.dispatcher, self.chunk,
                                  self.descriptor, rank=0)
        self.assertIn('already open', str(caught.exception))
        for name in ('qsa.0', 'qsa.1'):
            self.assertFalse(self.binding.holds[name].released)

    def test_the_same_frame_can_be_reused_after_the_lane_closed(self):
        registry = self.dispatcher.source_registry
        allocator = self.dispatcher.source_allocator
        for opened in self.binding.opened:
            opened.hold.release()
        self.binding.close(registry)
        again = ns.bind_local_sources(self.dispatcher, self.chunk,
                                      self.descriptor, rank=0)
        self.assertIsNot(again, self.binding)
        self.assertIs(again.descriptor, self.descriptor)
        self.assertEqual(again.layer_tables, self.binding.layer_tables)
        # A retry over the same live pool is the same allocation frame: the
        # registration records and their epochs do not move.
        self.assertIs(again.registrations['qsa.0'],
                      self.binding.registrations['qsa.0'])
        self.assertIs(again.registrations['qsa.1'],
                      self.binding.registrations['qsa.1'])
        self.assertEqual(again.source_epochs, self.binding.source_epochs)
        self.assertEqual(allocator.inventory()['live'], 2)
        self.assertEqual(allocator.inventory()['registrations'], 2)
        for opened in again.opened:
            opened.hold.release()
        again.close(registry)
        self.assertEqual(allocator.inventory()['live'], 0)

    def test_successive_chunk_rebinds_closed_registry_to_new_storage_epoch(self):
        registry = self.dispatcher.source_registry
        for opened in self.binding.opened:
            opened.hold.release()
        self.binding.close(registry)
        previous = self.binding.registrations['qsa.0']
        # WorkerState.reserve increments the prefill storage epoch on EVERY
        # chunk. A same-LaneId retry cannot represent a later chunk.
        self.dispatcher.operation.lane = replace(
            self.lane, storage_epoch=self.lane.storage_epoch + 1)
        tx = replace(self.admission.transaction, txn_seq=2)
        admission = replace(self.admission, transaction=tx, state_epoch=3)
        descriptor = make_descriptor(admission)
        again = ns.bind_local_sources(self.dispatcher, chunk_for(admission),
                                      descriptor, rank=0)
        self.assertIs(self.dispatcher.source_registry, registry)
        self.assertEqual(registry.lane, self.dispatcher.operation.lane)
        self.assertGreater(again.registrations['qsa.0'].mapping_version,
                           previous.mapping_version)
        self.assertEqual(again.source_epochs, self.binding.source_epochs)
        for opened in again.opened:
            opened.hold.release()
        again.close(registry)

    def test_lane_epoch_cannot_change_while_any_source_lease_remains(self):
        self.dispatcher.operation.lane = replace(
            self.lane, storage_epoch=self.lane.storage_epoch + 1)
        with self.assertRaises(shd.SourceRefused):
            ns.bind_local_sources(self.dispatcher, self.chunk,
                                  self.descriptor, rank=0)
        self.assertEqual(self.dispatcher.source_registry.lane, self.lane)
        self.assertEqual(self.dispatcher.source_allocator.inventory()['live'], 2)

    def test_a_stale_descriptor_is_refused_before_any_hold_exists(self):
        registry = self.dispatcher.source_registry
        allocator = self.dispatcher.source_allocator
        for opened in self.binding.opened:
            opened.hold.release()
        self.binding.close(registry)
        stale = replace(self.descriptor, state_epoch=self.admission.state_epoch + 1)
        with self.assertRaises(shd.SourceRefused) as caught:
            ns.bind_local_sources(self.dispatcher, self.chunk, stale, rank=0)
        self.assertIn('state epoch differs', str(caught.exception))
        self.assertEqual(registry._active, {})
        self.assertEqual(allocator.inventory()['live'], 0)

    def test_a_partial_open_rolls_back_every_retained_layer(self):
        runner = SourceRunner(pool=12)
        runner.sources['qsa.1'] = T((12, 1, PAGE_ROWS, ROW_WIDTH),
                                    device='cpu', dtype='torch.bfloat16')
        runner.layers[1]._qsa_host_kv = runner.sources['qsa.1']
        dispatcher = _Dispatcher(self.lane, runner)
        with self.assertRaises(ns.NativeSourceError) as caught:
            ns.bind_local_sources(dispatcher, chunk_for(self.admission),
                                  self.descriptor, rank=0)
        self.assertIn('not pinned', str(caught.exception))
        self.assertEqual(dispatcher.source_registry._active, {})
        self.assertEqual(dispatcher.source_allocator.inventory()['live'], 0)
        self.assertIsNone(getattr(dispatcher, 'source_binding', None))

    def test_a_null_or_repeated_admitted_page_is_refused(self):
        null_row = make_admission(block_tables=((0, 7, 9),))
        with self.assertRaises(shd.SourceRefused) as caught:
            ns.bind_local_sources(self._fresh(), chunk_for(null_row),
                                  w.SourceLeaseDescriptor(
                                      null_row.transaction, null_row.state_epoch,
                                      (7,), (0,)), rank=0)
        self.assertIn('null/invalid', str(caught.exception))
        # Two admitted pages are required here, so a repeated id is visible.
        repeated = make_admission(
            chunk_end=2 * PAGE_ROWS + 1, prompt_end=2 * PAGE_ROWS + 2,
            block_tables=((3, 3, 9),))
        with self.assertRaises(shd.SourceRefused) as caught:
            ns.bind_local_sources(self._fresh(), chunk_for(repeated),
                                  w.SourceLeaseDescriptor(
                                      repeated.transaction,
                                      repeated.state_epoch, (3, 9), (0,)), rank=0)
        self.assertIn('repeats a source page', str(caught.exception))

    def test_a_page_outside_the_pinned_pool_is_refused(self):
        admission = make_admission(block_tables=((40, 7, 9),))
        with self.assertRaises(shd.SourceRefused) as caught:
            ns.bind_local_sources(self._fresh(), chunk_for(admission),
                                  w.SourceLeaseDescriptor(
                                      admission.transaction,
                                      admission.state_epoch, (7, 9), (0,)),
                                  rank=0)
        self.assertIn('outside the pinned host pool', str(caught.exception))

    def test_a_changed_source_identity_bumps_its_epoch_and_mapping_version(self):
        registry = self.dispatcher.source_registry
        allocator = self.dispatcher.source_allocator
        first = self.binding.registrations['qsa.0']
        for opened in self.binding.opened:
            opened.hold.release()
        self.binding.close(registry)
        replacement = T((12, 1, PAGE_ROWS, ROW_WIDTH), device='cpu',
                        dtype='torch.bfloat16', pinned=True)
        self.runner.sources['qsa.0'] = replacement
        self.runner.layers[0]._qsa_host_kv = replacement
        again = ns.bind_local_sources(self.dispatcher, self.chunk,
                                      self.descriptor, rank=0)
        second = again.registrations['qsa.0']
        self.assertIs(second.source, replacement)
        self.assertGreater(second.source_epoch, first.source_epoch)
        self.assertGreater(second.mapping_version, first.mapping_version)
        self.assertGreater(second.source_incarnation, first.source_incarnation)
        self.assertEqual(again.source_epochs['qsa.0'], second.source_epoch)
        for opened in again.opened:
            opened.hold.release()
        again.close(registry)
        self.assertEqual(allocator.inventory()['live'], 0)

    def _fresh(self):
        return _Dispatcher(sparse_test.make_lane(), SourceRunner(pool=12))


# --- engine + bridge + driver construction ------------------------------------

class EnginePrepareSourceTests(unittest.TestCase):
    """The engine's actual descriptor on the first PREPARE, and its refusals."""

    def fixture(self):
        fixture = admission_test.DriverTests('test_one_bounded_command_per_tick')
        fixture.setUp()
        return fixture

    def drain_plan(self, fixture, first):
        commands = [first]
        for _ in range(40):
            fixture.executor._flash_s5_transport.reply(
                fixture.acceptances(commands[-1]))
            fixture.driver.advance()
            if fixture.driver.phase != 'segments':
                break
            _method, command = fixture.executor.last_command()
            commands.append(command)
        self.assertEqual(fixture.driver.phase, 'finalize')
        return commands

    def test_first_prepare_carries_the_actual_scheduler_hold(self):
        fixture = self.fixture()
        command = fixture.begin_segments()
        self.assertIs(command.kind, w.SegmentKind.PREPARE)
        hold = fixture.controller.source_hold(fixture.record.transaction)
        descriptor = hold.descriptor()
        self.assertIsNotNone(command.source)
        self.assertEqual(command.source, descriptor)
        self.assertEqual(command.source.block_ids, hold.pages)
        self.assertEqual(command.source.consumer_ranks,
                         tuple(sorted(fixture.controller.ranks)))
        self.assertEqual(command.source.transaction,
                         fixture.record.transaction)
        self.assertEqual(command.source.state_epoch, fixture.record.state_epoch)
        self.assertTrue(command.source.validate_frame(fixture.record, (0,)))
        self.assertEqual(hold.pages, hold.descriptor().block_ids)

    def test_only_prepare_commands_carry_the_lease(self):
        fixture = self.fixture()
        commands = self.drain_plan(fixture, fixture.begin_segments())
        prepares = [command for command in commands
                    if command.kind is w.SegmentKind.PREPARE]
        self.assertEqual(len(prepares), 2)
        hold = fixture.controller.source_hold(fixture.record.transaction)
        for command in commands:
            if command.kind is w.SegmentKind.PREPARE:
                self.assertEqual(command.source, hold.descriptor())
            else:
                self.assertIsNone(command.source)
        # A non-PREPARE command cannot be given a source retrospectively.
        other = next(command for command in commands
                     if command.kind is not w.SegmentKind.PREPARE)
        with self.assertRaises(et.E3ContractError) as caught:
            w.SegmentCommand(other.descriptor, other.op_seq, other.ranks,
                             source=hold.descriptor())
        self.assertIn('only a PREPARE', str(caught.exception))

    def test_prepare_before_the_lane_owners_are_bound_is_refused(self):
        fixture = self.fixture()
        fixture.driver.tx = fixture.record.transaction
        fixture.driver.admission = fixture.record
        with self.assertRaises(ed.DriverError) as caught:
            fixture.driver._prepare_source((0,))
        self.assertIn('owners are not bound', str(caught.exception))

    def test_prepare_for_a_foreign_transaction_has_no_hold(self):
        fixture = self.fixture()
        fixture.begin_segments()
        foreign = replace(fixture.record.transaction,
                          txn_seq=fixture.record.transaction.txn_seq + 1)
        fixture.driver.tx = foreign
        with self.assertRaises(et.E3ContractError) as caught:
            fixture.driver._prepare_source((0,))
        self.assertIn('source hold belongs to another transaction',
                      str(caught.exception))

    def test_prepare_with_a_stale_admission_epoch_is_refused(self):
        fixture = self.fixture()
        fixture.begin_segments()
        fixture.driver.admission = replace(
            fixture.record, state_epoch=fixture.record.state_epoch + 1)
        with self.assertRaises(ed.DriverError) as caught:
            fixture.driver._prepare_source((0,))
        self.assertIn('state epoch is stale', str(caught.exception))

    def test_prepare_for_a_rank_outside_the_lease_is_refused(self):
        fixture = self.fixture()
        fixture.begin_segments()
        with self.assertRaises(ed.DriverError) as caught:
            fixture.driver._prepare_source((2,))
        self.assertIn('does not name every consumer rank',
                      str(caught.exception))

    def test_a_widened_or_foreign_descriptor_cannot_authorise_the_frame(self):
        fixture = self.fixture()
        fixture.begin_segments()
        hold = fixture.controller.source_hold(fixture.record.transaction)
        widened = replace(hold.descriptor(),
                          block_ids=tuple(sorted(hold.pages + (999,))))
        with self.assertRaises(et.E3ContractError) as caught:
            widened.validate_frame(fixture.record, (0,))
        self.assertIn('outside the admitted rows', str(caught.exception))
        with self.assertRaises(et.E3ContractError) as caught:
            hold.descriptor().validate_frame(fixture.record, (2,))
        self.assertIn('does not name every consumer rank',
                      str(caught.exception))

    def test_a_retained_cow_source_page_is_an_admitted_lease_page(self):
        admission = make_admission(block_tables=((3, 7, 9),),
                                   cow_pairs=((11, 12),))
        admitted = w.SourceLeaseDescriptor(
            admission.transaction, admission.state_epoch, (3, 11, 12), (0,))
        self.assertTrue(admitted.validate_frame(admission, (0,)))
        # A CoW page that was never named by the frame remains foreign.
        foreign = w.SourceLeaseDescriptor(
            admission.transaction, admission.state_epoch, (3, 11, 13), (0,))
        with self.assertRaises(et.E3ContractError) as caught:
            foreign.validate_frame(admission, (0,))
        self.assertIn('outside the admitted rows', str(caught.exception))


# --- the worker server boundary ------------------------------------------------

class SourceServer:
    """Worker-side server over the real construction, with no layer executed.

    ``begin_prefill_operation`` is the reviewed native binding path.  This
    fixture's lane never invokes the attention driver, so there is no honest
    way to call ``end_forward``; the terminal step is an explicit cancellation
    recorded as such, never a fabricated layer completion.
    """

    def __init__(self, state, runner, qsa_module, device):
        self.runtime = r4.FakeRuntime(state)
        self.runner = runner
        self.qsa_module = qsa_module
        self.operation_device = device
        self.operation = None
        self.transaction = None
        self._metadata_snapshot = None
        self._installed = ()
        self.calls = []
        self.worker = None

    def attach_worker(self, worker):
        self.worker = worker
        return self

    def open(self, message):
        if self.operation is not None:
            raise AssertionError('worker server opened twice')
        state = self.runtime.state
        state.declare_request(message.transaction.request_uid,
                              message.transaction.request_generation)
        state.prefill.reserve(message.transaction, metadata=message.metadata)
        state.prefill.ready()
        lane = state.prefill.handle.lane_id
        try:
            provider = qo.metadata_plan_provider(message.metadata)
        except qo.OperationError:
            provider = None
        self.operation = qo.QsaPrefillOperation(
            message.transaction, lane, device=self.operation_device,
            plan_provider=provider, clock=lambda: 7)
        self.transaction = message.transaction
        self._metadata_snapshot = message.metadata
        self.calls.append(('open', message.transaction, lane))
        return types.SimpleNamespace(transaction=message.transaction,
                                     accepted=True, lane=None)

    def begin_prefill_operation(self, runner, marker):
        operation = self.operation
        if operation is None:
            raise AssertionError('no operation is bound')
        if self._installed:
            raise AssertionError('operation is already installed')
        if marker.flash_e3_transaction != operation.transaction:
            raise AssertionError('marker transaction differs')
        layers = native_resources._qsa_layers(runner)
        native_resources.bind_operation_sources(
            operation, layers, marker.flash_e3_source_binding,
            self._metadata_snapshot)
        operation.begin_forward(layers=tuple(name for name, _ in layers))
        for name, layer in layers:
            if getattr(layer, '_flash_s5_operation', None) is not None:
                raise AssertionError('layer already carries an operation: ' + name)
            layer._flash_s5_operation = operation
        self._installed = tuple(layers)
        self.calls.append(('begin', operation.transaction,
                           tuple(name for name, _ in layers)))
        return operation

    def _uninstall(self):
        installed, self._installed = self._installed, ()
        for _name, layer in installed:
            if getattr(layer, '_flash_s5_operation', None) is self.operation:
                delattr(layer, '_flash_s5_operation')
        return bool(installed)

    def end_prefill_operation(self, runner):
        operation = self.operation
        self._uninstall()
        operation.cancel('construction fixture: no native layer executed')
        self.calls.append(('end-cancelled', operation.transaction))
        return operation

    def abort_prefill_operation(self, runner):
        operation = self.operation
        self._uninstall()
        operation.cancel('construction fixture: aborted before any layer')
        self.calls.append(('abort-cancelled', operation.transaction))
        return operation

    def clear_native(self, transaction):
        if self.transaction != transaction:
            raise AssertionError('clear_native for another transaction')
        self.operation = None
        self.transaction = None
        self._metadata_snapshot = None
        self.calls.append(('clear', transaction))
        return True


def source_worker(rank=0, stage=0, *, pool=8):
    """One real dispatcher/bridge/driver over fake streams, events and tensors."""
    bus = r4.EventBus()
    torch = r4.bus_torch(bus)
    state = r4.WorkerState(rank, engine_epoch=0, engine_bound=False)
    state.retain_decode_storage(context_id=100 + rank)
    split, controller, router, route_backend = r4.route_objects(
        rank, state, bus, torch)
    channel = r4.FakePPChannel(bus, is_sender=stage == 0)
    lane = r4.FakeLane(layers=(0, 1) if stage == 0 else (2, 3))
    lane.is_sender = stage == 0
    runner = SourceRunner(pool=pool)
    device = sparse_test.FakeSparseDevice(auto_ready=True)
    qsa_module = types.SimpleNamespace(
        _flash_s5_staged_operation=lambda *args, **kwargs: None,
        _flash_s5_sparse_staged_operation=lambda *args, **kwargs: None)

    def driver_factory(operation, module, context):
        return qsa_driver.QsaLayerDriver(
            operation,
            staged_operation=module._flash_s5_staged_operation,
            sparse_staged_operation=module._flash_s5_sparse_staged_operation,
            forward_context=lambda: None,
            source_holds=getattr(operation, 'native_source_holds', None))

    native = types.SimpleNamespace(
        split=split,
        comms=types.SimpleNamespace(pp=channel),
        copy_stream=r4.FakeStream(1000 + rank),
        controller=controller,
        router=router,
        lane_forward=lambda inputs, forward_context=None,
        attention_driver=None: lane)
    server = SourceServer(state, runner, qsa_module, device)
    factory = r4.FakeFactory(rank)
    dispatcher = lane_dispatch.LaneDispatcher(
        runner, native, factory, server=server,
        qsa_module=qsa_module, driver_factory=driver_factory,
        forward_context_getter=lambda: None, torch_module=torch,
        clock=lambda: 1000 + rank)
    bridge = worker_bridge.NativeWorkerBridge(dispatcher, server=server)
    worker = r4.FakeWorker(rank, stage, bridge, dispatcher)
    worker.bus = bus
    worker.torch = torch
    worker.state = state
    worker.split = split
    worker.controller = controller
    worker.router = router
    worker.route_backend = route_backend
    worker.lane = lane
    worker.channel = channel
    worker.server = server
    worker.factory = factory
    worker.runner = runner
    worker.device = device
    bridge.attach(worker)
    return worker


class SourceStack(r4.Stack):
    """The reviewed four-worker stack with rank 0 running the native path."""

    def _add_worker(self, rank, stage):
        if rank != 0:
            return super()._add_worker(rank, stage)
        worker = source_worker(0, stage)
        self.workers[0] = worker
        self.source_worker = worker


class BridgeConstructionTests(unittest.TestCase):
    """Real engine PREPARE -> real bridge -> real QsaLayerDriver construction."""

    def setUp(self):
        self.stack = SourceStack()
        self.stack.drive_to_finalize()
        self.worker = self.stack.source_worker
        self.dispatcher = self.worker.dispatcher
        self.bridge = self.worker._flash_s5_bridge
        self.binding = self.dispatcher.source_binding
        self.operation = self.dispatcher.operation

    def wire_prepare(self, local):
        """The wire command the engine actually submitted, rebuilt from its
        accepted local view (the local view keeps no ranks)."""
        descriptor = w.SegmentDescriptor(
            local.transaction, w.SegmentKind(local.kind), local.stage,
            layer_start=local.layer_start, layer_end=local.layer_end,
            seq=local.seq)
        return w.SegmentCommand(descriptor, local.op_seq, (0, 1),
                                source=local.source)

    def prepared(self):
        return next(command for command, _ack in
                    self.dispatcher.accepted.values()
                    if command.kind == 'prepare')

    def test_the_first_prepare_binds_the_leases_and_builds_the_layer_driver(self):
        self.assertIsNotNone(self.binding)
        self.assertIs(type(self.binding), ns.NativeSourceBinding)
        self.assertEqual(self.binding.layers, ('qsa.0', 'qsa.1'))
        self.assertEqual(self.binding.rank, 0)
        self.assertEqual(self.binding.transaction,
                         self.stack.record.transaction)
        self.assertEqual(self.binding.state_epoch,
                         self.stack.record.state_epoch)
        # The worker proved the engine's actual descriptor, not a local copy.
        prepare = self.prepared()
        hold = self.stack.controller.source_hold(self.stack.record.transaction)
        self.assertEqual(prepare.source, hold.descriptor())
        self.assertTrue(prepare.source.validate_frame(
            self.dispatcher.admission, (0,)))
        # Every other accepted segment carries no source at all.
        for command, _ack in self.dispatcher.accepted.values():
            if command.kind != 'prepare':
                self.assertIsNone(command.source)
        # The real driver was constructed over the real operation and holds.
        driver = self.dispatcher.driver
        self.assertIs(type(driver), qsa_driver.QsaLayerDriver)
        self.assertIs(driver.operation, self.operation)
        self.assertEqual(set(driver.source_holds), set(self.binding.holds))
        for name, hold in self.binding.holds.items():
            self.assertIs(driver.source_holds[name], hold)
        self.assertIs(driver.sparse_staged_operation,
                      self.worker.server.qsa_module
                      ._flash_s5_sparse_staged_operation)
        self.assertEqual(self.operation.native_source_holds,
                         self.binding.holds)
        self.assertEqual(self.operation.state_epoch,
                         self.stack.record.state_epoch)
        self.assertEqual(self.operation.plan_provider('qsa.0', 1), (1,))
        # Both layers were registered and retained on the worker before any
        # segment could submit.
        registry = self.dispatcher.source_registry
        allocator = self.dispatcher.source_allocator
        self.assertEqual(len(registry._active), 2)
        self.assertEqual(allocator.inventory()['live'], 2)
        for hold in self.binding.holds.values():
            self.assertFalse(hold.released)
        names = ('begin', self.stack.record.transaction, ('qsa.0', 'qsa.1'))
        self.assertIn(names, self.worker.server.calls)
        self.assertIn(('end-cancelled', self.stack.record.transaction),
                      self.worker.server.calls)
        # The construction fixture executed no native layer, so the honest
        # terminal outcome is the recorded cancellation, not a pass.
        self.assertEqual(self.operation.cancelled,
                         'construction fixture: no native layer executed')
        for layer in self.worker.runner.layers:
            self.assertIsNone(getattr(layer, '_flash_s5_operation', None))

    def test_the_bridge_is_idempotent_for_the_same_lease_and_refuses_another(self):
        prepare = self.wire_prepare(self.prepared())
        before = len(self.dispatcher.accepted)
        accepted = self.bridge.submit(prepare)
        self.assertTrue(accepted.accepted)
        self.assertEqual(len(self.dispatcher.accepted), before)
        self.assertIs(self.dispatcher.source_binding, self.binding)
        # A different but individually valid lease is not silently re-bound.
        widened = replace(prepare.source, block_ids=(1, 3))
        with self.assertRaises(worker_bridge.WorkerBridgeError) as caught:
            self.bridge.submit(replace(prepare, source=widened))
        self.assertIn('different source lease', str(caught.exception))
        missing = replace(prepare, source=None)
        with self.assertRaises(worker_bridge.WorkerBridgeError) as caught:
            self.bridge.submit(missing)
        self.assertIn('missing the already-bound', str(caught.exception))
        self.assertFalse(self.binding.closed)

    def drive_to_release(self):
        for _ in range(300):
            self.stack.complete_everything()
            if self.stack.driver.tx is None:
                return
            self.stack.driver.advance()
        self.fail('the engine did not release the transaction')

    def test_release_closes_the_binding_only_after_every_hold_released(self):
        binding, operation = self.binding, self.operation
        registry = self.dispatcher.source_registry
        allocator = self.dispatcher.source_allocator
        self.assertFalse(binding.closed)
        self.drive_to_release()
        self.assertIsNone(self.stack.driver.tx)
        self.assertTrue(binding.closed)
        for name, hold in binding.holds.items():
            self.assertTrue(hold.released, name)
        self.assertEqual(allocator.inventory()['live'], 0)
        self.assertEqual(registry._active, {})
        self.assertIsNone(self.dispatcher.source_binding)
        self.assertIsNone(self.bridge._source_binding)
        self.assertTrue(operation._released)
        self.assertEqual(operation.stats['sparse_source_released'], 2)
        self.assertEqual(len(operation._source_release), 2)
        self.assertIn(('clear', self.stack.record.transaction),
                      self.worker.server.calls)
        self.assertEqual(self.worker.server.runtime.releases,
                         [self.stack.record.transaction])


class BridgeRefusalTests(unittest.TestCase):
    """The bridge refuses before dispatch; the engine's lease is never invented."""

    def setUp(self):
        self.worker = source_worker(0, 0)
        self.bridge = self.worker._flash_s5_bridge
        self.dispatcher = self.worker.dispatcher
        self.admission = make_admission(block_tables=((1, 2, 3, 4),),
                                        prompt_end=6)
        # Direct worker fixture: the engine owner is declared here instead of
        # arriving through the driver's bootstrap phase.
        self.worker.state.bind_engine_owner(
            self.admission.transaction.engine_epoch)
        self.request = w.OpenRequest(self.admission, (0,),
                                     self.admission.state_epoch, 1)
        self.bridge.open(self.request)

    def tearDown(self):
        operation = self.worker.server.operation
        if operation is not None and not operation._released:
            operation.cancel('fixture teardown')
            operation.release()

    def command(self, source):
        descriptor = w.SegmentDescriptor(
            self.admission.transaction, w.SegmentKind.PREPARE, 0)
        return w.SegmentCommand(descriptor, 0, (0,), source=source)

    def descriptor(self, **overrides):
        values = dict(transaction=self.admission.transaction,
                      state_epoch=self.admission.state_epoch,
                      block_ids=(1,), consumer_ranks=(0,))
        values.update(overrides)
        return w.SourceLeaseDescriptor(**values)

    def test_a_prepare_without_the_lease_is_refused_before_any_submission(self):
        with self.assertRaises(worker_bridge.WorkerBridgeError) as caught:
            self.bridge.submit(self.command(None))
        self.assertIn('prepare carries no source lease', str(caught.exception))
        self.assertIsNone(self.dispatcher.source_binding)
        self.assertEqual(len(self.dispatcher.accepted), 0)

    def test_a_stale_or_foreign_lease_is_refused_before_any_submission(self):
        stale = self.descriptor(
            state_epoch=self.admission.state_epoch + 1)
        with self.assertRaises(worker_bridge.WorkerBridgeError) as caught:
            self.bridge.submit(self.command(stale))
        self.assertIn('state epoch is stale', str(caught.exception))
        foreign = self.descriptor(transaction=replace(
            self.admission.transaction,
            txn_seq=self.admission.transaction.txn_seq + 1))
        # Wire-v1 refuses to build a command whose lease names another
        # transaction, so the bridge's own source check is exercised directly.
        with self.assertRaises(worker_bridge.WorkerBridgeError) as caught:
            self.bridge._bind_prepare_source(
                types.SimpleNamespace(source=foreign))
        self.assertIn('another transaction', str(caught.exception))
        outside = self.descriptor(block_ids=(999,))
        with self.assertRaises(worker_bridge.WorkerBridgeError) as caught:
            self.bridge.submit(self.command(outside))
        self.assertIn('source lease was refused', str(caught.exception))
        # Wire-v1 itself refuses a command whose target rank the lease omits.
        wrong_rank = self.descriptor(consumer_ranks=(1,))
        with self.assertRaises(et.E3ContractError) as caught:
            self.command(wrong_rank)
        self.assertIn('does not name every target rank', str(caught.exception))
        self.assertIsNone(self.dispatcher.source_binding)
        self.assertIsNone(self.dispatcher.operation.native_source_holds
                          if hasattr(self.dispatcher.operation,
                                     'native_source_holds') else None)

    def test_the_good_lease_binds_once_and_can_then_be_released(self):
        command = self.command(self.descriptor())
        accepted = self.bridge.submit(command)
        self.assertTrue(accepted.accepted)
        binding = self.dispatcher.source_binding
        self.assertIsNotNone(binding)
        self.assertIs(self.dispatcher.driver.__class__,
                      qsa_driver.QsaLayerDriver)
        self.assertEqual(set(binding.holds), {'qsa.0', 'qsa.1'})
        self.assertIn(('begin', self.admission.transaction,
                       ('qsa.0', 'qsa.1')), self.worker.server.calls)
        # Retrying the identical lease is inert; no second binding is made.
        again = self.bridge.submit(command)
        self.assertTrue(again.accepted)
        self.assertIs(self.dispatcher.source_binding, binding)


# --- operation binding (cancel before a layer runs) ---------------------------

class OperationBindingTests(unittest.TestCase):
    """``bind_operation_sources`` retains the admitted holds at construction."""

    def fixture(self):
        runner = SourceRunner(pool=12)
        lane = sparse_test.make_lane()
        dispatcher = _Dispatcher(lane, runner)
        admission = make_admission()
        descriptor = w.SourceLeaseDescriptor(
            admission.transaction, admission.state_epoch, (3, 7, 9), (0,))
        binding = ns.bind_local_sources(
            dispatcher, chunk_for(admission), descriptor, rank=0)
        operation = qo.QsaPrefillOperation(
            admission.transaction, lane,
            device=sparse_test.FakeSparseDevice(auto_ready=True),
            plan_provider=None, clock=lambda: 7)
        snapshot = worker_bridge.admission_metadata(admission)
        self.addCleanup(self.close, dispatcher, binding)
        return runner, lane, dispatcher, admission, descriptor, binding, \
            operation, snapshot

    def close(self, dispatcher, binding):
        for opened in binding.opened:
            if not opened.hold.released:
                opened.hold.release()
        if not binding.closed:
            binding.close(dispatcher.source_registry)

    def variant(self, binding, **overrides):
        values = dict(transaction=binding.transaction,
                      state_epoch=binding.state_epoch, rank=binding.rank,
                      holds=binding.holds, registrations=binding.registrations,
                      opened=binding.opened, layer_tables=binding.layer_tables,
                      group_rows=binding.group_rows, sources=binding.sources,
                      source_epochs=binding.source_epochs,
                      versions=binding.versions)
        values.update(overrides)
        return ns.NativeSourceBinding(binding.descriptor, **values)

    def test_binding_attaches_every_hold_and_cancel_releases_them(self):
        runner, lane, dispatcher, admission, descriptor, binding, \
            operation, snapshot = self.fixture()
        layers = native_resources._qsa_layers(runner)
        operation.native_source_holds = binding.holds
        native_resources.bind_operation_sources(
            operation, layers, binding, snapshot)
        self.assertEqual(set(operation._layers), {'qsa.0', 'qsa.1'})
        for name in ('qsa.0', 'qsa.1'):
            self.assertIs(operation._layers[name].sparse_hold,
                          binding.holds[name])
            self.assertEqual(operation._layers[name].state_epoch,
                             admission.state_epoch)
        operation.begin_forward(layers=('qsa.0', 'qsa.1'))
        # No layer ran; the recorded cancellation is the honest outcome.
        operation.cancel('cancelled before any layer')
        operation.release()
        for name, hold in binding.holds.items():
            self.assertTrue(hold.released, name)
        self.assertEqual(dispatcher.source_allocator.inventory()['live'], 0)
        binding.close(dispatcher.source_registry)
        self.assertTrue(binding.closed)
        self.assertEqual(dispatcher.source_registry._active, {})

    def test_a_missing_hold_refuses_before_any_forward(self):
        runner, lane, dispatcher, admission, descriptor, binding, \
            operation, snapshot = self.fixture()
        layers = native_resources._qsa_layers(runner)
        # The layer set is still complete; only the admitted hold is absent.
        missing = dict(binding.holds)
        missing['qsa.1'] = None
        with self.assertRaises(native_resources.HandoffError) as caught:
            native_resources.bind_operation_sources(
                operation, layers, self.variant(binding, holds=missing),
                snapshot)
        self.assertIn('has no hold for qsa.1', str(caught.exception))
        self.assertFalse(operation._forward_open)

    def test_a_foreign_source_tensor_refuses_before_any_forward(self):
        runner, lane, dispatcher, admission, descriptor, binding, \
            operation, snapshot = self.fixture()
        layers = native_resources._qsa_layers(runner)
        # The real source tensor identity is proved, not just the layer name.
        sources = dict(binding.sources)
        sources['qsa.0'] = T((12, 1, PAGE_ROWS, ROW_WIDTH), device='cpu',
                             dtype='torch.bfloat16', pinned=True)
        with self.assertRaises(native_resources.HandoffError) as caught:
            native_resources.bind_operation_sources(
                operation, layers, self.variant(binding, sources=sources),
                snapshot)
        self.assertIn('does not own this layer history', str(caught.exception))
        self.assertFalse(operation._forward_open)

    def test_a_fragment_cap_is_a_typed_serial_refusal(self):
        # The compact plan refuses the whole selection before a single copy;
        # this is the unsupported admission outcome, never an overlap pass.
        selected = tuple(range(0, 1024, 2))
        mapping = tuple((index * 17 + 3) % 52 for index in range(50))
        versions = tuple(range(64))
        plan = shd.make_sparse_plan(selected, mapping, versions, 1)
        self.assertTrue(plan.refused)
        self.assertEqual(plan.fallback.reason, 'descriptor_cap')
        with self.assertRaises(shd.SparsePlanRefused) as caught:
            shd.validate_plan_source(plan, mapping, versions, 1)
        self.assertEqual(caught.exception.refusal.reason, 'descriptor_cap')
        self.assertEqual(caught.exception.refusal.epoch, 1)


class DriverRefusalTests(unittest.TestCase):
    """``_bucket`` turns a typed refusal into an explicit stopped transaction.

    Neither outcome is counted as an overlap bucket, so a fragment cap or a
    byte mismatch can never be recorded as a sparse success.
    """

    class Operation:
        def __init__(self, effect):
            self.effect = effect
            self.cancelled = []

        def advance(self):
            raise self.effect

        def cancel(self, reason):
            self.cancelled.append(reason)
            return reason

    def driver(self, effect):
        operation = self.Operation(effect)
        driver = qsa_driver.QsaLayerDriver(
            operation, staged_operation=lambda *args: None,
            sparse_staged_operation=lambda *args: None,
            forward_context=lambda: None)
        return driver, operation

    def phase(self):
        return qsa_driver.LayerPhase(0, None, None, None)

    def test_a_row1_oracle_failure_aborts_and_is_not_an_overlap_bucket(self):
        message = 'row1 qualification failed: arena_bytes'
        driver, operation = self.driver(ro.Row1OracleError(message))
        with self.assertRaises(qsa_driver.QsaDriverError) as caught:
            driver._bucket(None, self.phase())
        self.assertIn('native row1 qualification failed', str(caught.exception))
        self.assertEqual(driver.records[-1],
                         ('row1_failure', 0, message))
        self.assertEqual(operation.cancelled, ['row1_failure'])
        self.assertNotIn('bucket', [record[0] for record in driver.records])

    def test_a_fragment_cap_is_a_typed_serial_refusal(self):
        refusal = shd.SparseRefusal('descriptor_cap', 0, 0, 5, 2, 10240, 1)
        for effect in (shd.SparsePlanRefused(refusal),
                       shd.SparseFallbackRequired(refusal)):
            driver, operation = self.driver(effect)
            with self.assertRaises(qsa_driver.QsaDriverError) as caught:
                driver._bucket(None, self.phase())
            self.assertIn('sparse history refused the overlap route',
                          str(caught.exception))
            self.assertEqual(driver.records[-1],
                             ('sparse_refusal', 0, 'descriptor_cap', 5, 10240))
            self.assertEqual(operation.cancelled, ['sparse_refusal'])
            self.assertNotIn('bucket',
                             [record[0] for record in driver.records])


# --- the E2 snapshot oracle over fake bytes -----------------------------------

class RowBackend:
    """The real snapshot/gather surface over the fixture's fake byte storage."""

    def __init__(self, *, gather_mutation=None):
        self.gather_mutation = gather_mutation

    def snapshot(self, tensor):
        storage = getattr(tensor, 'storage', None)
        if storage is not None:
            return tuple(storage.rows)
        if getattr(tensor, 'values', None) is not None:
            return tuple(tensor.values)
        raise AssertionError('row1 snapshot needs byte rows or values')

    def equal(self, first, second):
        return first == second

    def nbytes(self, snapshot):
        # Byte payloads count their actual length; a value row is one bf16.
        total = 0
        for item in snapshot:
            if item is None:
                continue
            total += len(item) if isinstance(item, (bytes, bytearray)) else 2
        return total

    def differing_bytes(self, first, second):
        return (sum(1 for left, right in zip(first, second) if left != right)
                + abs(len(first) - len(second)))

    def finite(self, snapshot):
        for item in snapshot:
            if item is None:
                return False
            if isinstance(item, float) and not math.isfinite(item):
                return False
        return True

    def gather_rows(self, host, source_row, rows, staged, arena_row):
        storage = staged.storage
        if storage is None:
            raise AssertionError('row1 gather needs arena row storage')
        for offset in range(rows):
            row = source_row + offset
            if self.gather_mutation is not None and row == self.gather_mutation:
                payload = b'MUTATED'
            else:
                payload = host.row_bytes(row)
            storage.rows[staged.row_offset + arena_row + offset] = payload

    def gather_reference(self, host, table, lo, hi, staged):
        storage = staged.storage
        if storage is None:
            raise AssertionError('reference gather needs arena row storage')
        for page in range(lo, hi):
            source_page = int(table.values[page])
            for offset in range(PAGE_ROWS):
                storage.rows[staged.row_offset
                             + (page - lo) * PAGE_ROWS + offset] = \
                    host.row_bytes(source_page * PAGE_ROWS + offset)


class Row1QualifierTests(unittest.TestCase):
    """Byte-for-byte bucket/output comparisons; digests only bind."""

    def setUp(self):
        self.lane = sparse_test.make_lane()
        self.transaction = sparse_test.make_transaction(chunk_end=8096)
        self.store = ro.Row1RecordStore(capacity=32)
        self.backend = RowBackend()
        self.epochs = dict(worker_incarnation='worker-7', engine_epoch=3,
                           route_generation=2)
        self.qualifier = ro.Row1Qualifier(
            self.store, transaction=self.transaction, lane=self.lane,
            backend=self.backend, epochs=lambda: dict(self.epochs))
        self.uva = sparse_test.T((12, 1, PAGE_ROWS, ROW_WIDTH), device='cpu',
                                 dtype='torch.bfloat16', pinned=True)
        self.arena = sparse_test.T(
            (ARENA_PAGES, 1, PAGE_ROWS, ROW_WIDTH), device='cuda:0',
            dtype='torch.bfloat16', storage=Rows(ARENA_ROWS))

    def make_run(self, **overrides):
        values = dict(layer_name='qsa.0', allocation_epoch=3, source_epoch=1,
                      state_epoch=2, mapping_version=4, uva=self.uva,
                      row1_failed=None, sparse_hold=None,
                      row1_reference=None)
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def bucket(self, *, index=0, segments=((12, 0, 5), (40, 5, 3))):
        rows = sum(rows for _source, _arena, rows in segments)
        return ro.Row1Bucket(index=index, route='sparse_dma', rows=rows,
                             pieces=len(segments), compact_start=0,
                             nbytes=rows * ROW_BYTES, segments=segments)

    def fill_actual(self, bucket):
        for source_row, arena_row, rows in bucket.segments:
            for offset in range(rows):
                self.arena.storage.rows[arena_row + offset] = \
                    self.uva.row_bytes(source_row + offset)

    def test_a_matching_bucket_records_the_actual_bytes(self):
        run = self.make_run()
        bucket = self.bucket()
        self.fill_actual(bucket)
        record = self.qualifier.bucket(run, bucket, self.arena)
        self.assertTrue(record.equal)
        self.assertIsNone(record.different_bytes)
        self.assertEqual(record.identity.rank, 0)
        self.assertEqual(record.identity.worker_incarnation, 'worker-7')
        self.assertEqual(record.identity.engine_epoch, 3)
        self.assertEqual(record.identity.route_generation, 2)
        self.assertEqual(record.identity.storage_epoch, 3)
        self.assertEqual(record.identity.source_epoch, 1)
        self.assertEqual(record.identity.state_epoch, 2)
        self.assertEqual(record.identity.mapping_version, 4)
        self.assertEqual(record.identity.layer, 'qsa.0')
        self.assertEqual(record.identity.operation, 'sparse_row_bucket')
        self.assertEqual(record.coverage_rows, bucket.rows)
        self.assertEqual(record.coverage_bytes, bucket.rows * 32)
        self.assertEqual(len(record.digest), 64)
        self.assertEqual(len(self.store.records()), 1)
        self.assertEqual(self.store.failures(), ())
        self.assertEqual(self.qualifier.stats['buckets'], 1)
        # The digest binds the record; the equality above is the evidence.
        self.assertEqual(record.as_dict()['equal'], True)

    def test_a_mutated_arena_byte_fails_with_the_first_immutable_record(self):
        run = self.make_run(row1_reference=lambda: None)
        bucket = self.bucket()
        self.fill_actual(bucket)
        self.backend.gather_mutation = bucket.segments[0][0] + 2
        with self.assertRaises(ro.Row1OracleError) as caught:
            self.qualifier.bucket(run, bucket, self.arena)
        self.assertIn('arena_bytes', str(caught.exception))
        self.assertEqual(run.row1_failed, 'arena_bytes')
        failures = self.store.failures()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].kind, 'arena_bytes')
        self.assertGreaterEqual(failures[0].different_bytes, 1)
        self.assertFalse(failures[0].hold_released)
        self.assertIn(('route', 'sparse_dma'), failures[0].evidence)
        records = self.store.records()
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].equal)
        self.assertEqual(records[0].different_bytes, 1)
        self.assertEqual(self.qualifier.stats['failures'], 1)
        # The failed run can never be released as a pass.
        with self.assertRaises(ro.Row1OracleError):
            self.qualifier.check_releasable(run)

    def test_bucket_segments_must_cover_exactly_its_rows(self):
        with self.assertRaises(ro.Row1OracleError):
            ro.Row1Bucket(index=0, route='sparse_dma', rows=9, pieces=1,
                          compact_start=0, nbytes=9 * ROW_BYTES,
                          segments=((1, 0, 5),))
        with self.assertRaises(ro.Row1OracleError):
            ro.Row1Bucket(index=0, route='sparse_uva', rows=5, pieces=1,
                          compact_start=0, nbytes=5 * ROW_BYTES,
                          segments=((1, 0, 5),))

    def test_identity_refuses_unbound_epochs_and_missing_owners(self):
        run = self.make_run(state_epoch=None)
        with self.assertRaises(ro.Row1OracleError) as caught:
            self.qualifier.bucket(run, self.bucket(), self.arena)
        self.assertIn('no admitted state epoch', str(caught.exception))
        for values in (dict(engine_epoch=0), dict(worker_incarnation='')):
            epochs = dict(self.epochs)
            epochs.update(values)
            qualifier = ro.Row1Qualifier(
                self.store, transaction=self.transaction, lane=self.lane,
                backend=self.backend, epochs=lambda: dict(epochs))
            with self.assertRaises(ro.Row1OracleError):
                qualifier.identity(self.make_run(), 'sparse_row_bucket')
        with self.assertRaises(ro.Row1OracleError):
            ro.Row1Identity(0, 'worker', 0, 0, 3, 0, 1, 1, 'qsa.0', 'op')
        with self.assertRaises(ro.Row1OracleError):
            ro.Row1Identity(0, '', 1, 0, 3, 0, 1, 1, 'qsa.0', 'op')
        # An integer worker incarnation is the state's own opaque identity and
        # zero is not an unbound sentinel there.
        self.assertEqual(ro.Row1Identity(0, 0, 1, 0, 3, 0, 1, 1,
                                         'qsa.0', 'op').worker_incarnation, 0)

    def output(self, run, reference, tensor, *, expected_route='sparse_uva',
               expected_fills=0):
        session = self.qualifier.reference_session(run)

        def original():
            # The operation binds the original function and the session; the
            # qualifier calls that zero-argument closure.
            return reference(session)

        run.row1_reference = original
        return self.qualifier.output(run, session, original, tensor,
                                     expected_route=expected_route,
                                     expected_fills=expected_fills)

    def out_tensor(self, values):
        return sparse_test.T((len(values), 1), device='cuda:0',
                             dtype='torch.bfloat16', values=list(values))

    def test_a_matching_output_records_the_before_moe_bytes(self):
        run = self.make_run()
        tensor = self.out_tensor([0.0, 0.0, 0.0, 0.0])
        calls = []

        def reference(session):
            self.assertIs(type(session), ro.Row1ReferenceSession)
            session.route('sparse_uva', 15)
            calls.append(('reference',))

        record = self.output(run, reference, tensor)
        self.assertTrue(record.equal)
        self.assertTrue(record.finite)
        self.assertEqual(record.identity.operation, 'qsa_output_before_moe')
        self.assertEqual(record.route, 'sparse_dma')
        self.assertEqual(record.rows, 4)
        self.assertEqual(record.pieces, 0)
        self.assertEqual(record.coverage_rows, 4)
        self.assertEqual(calls, [('reference',)])
        self.assertEqual(self.qualifier.stats['outputs'], 1)

    def test_a_mutated_output_byte_fails(self):
        run = self.make_run()
        tensor = self.out_tensor([0.0, 0.0, 0.0, 0.0])

        def reference(session):
            session.route('sparse_uva', 15)
            tensor.values[1] = 1.0

        with self.assertRaises(ro.Row1OracleError) as caught:
            self.output(run, reference, tensor)
        self.assertIn('qsa_output_bytes', str(caught.exception))
        self.assertEqual(run.row1_failed, 'qsa_output_bytes')
        failure = self.store.failures()[0]
        self.assertEqual(failure.kind, 'qsa_output_bytes')
        self.assertEqual(failure.different_bytes, 1)
        self.assertIn(('route', 'sparse_dma'), failure.evidence)

    def test_a_route_named_but_wrong_or_missing_is_refused(self):
        run = self.make_run()
        tensor = self.out_tensor([0.0, 0.0])

        def wrong(session):
            session.route('dense_dma', 3)

        with self.assertRaises(ro.Row1OracleError) as caught:
            self.output(run, wrong, tensor)
        self.assertIn('route_mismatch', str(caught.exception))
        self.assertEqual(self.store.failures()[0].kind, 'route_mismatch')
        run = self.make_run()
        tensor = self.out_tensor([0.0, 0.0])

        def missing(session):
            return None

        with self.assertRaises(ro.Row1OracleError) as caught:
            self.output(run, missing, tensor)
        self.assertIn('missing_reference_route', str(caught.exception))

    def test_a_reference_bucket_shortfall_is_refused(self):
        run = self.make_run()
        tensor = self.out_tensor([0.0, 0.0])

        def dense(session):
            session.route('dense_dma', 4)

        with self.assertRaises(ro.Row1OracleError) as caught:
            self.output(run, dense, tensor, expected_route='dense_dma',
                        expected_fills=2)
        self.assertIn('reference_bucket_coverage', str(caught.exception))
        self.assertEqual(self.store.failures()[0].kind,
                         'reference_bucket_coverage')

    def test_a_reference_exception_and_a_nonfinite_output_fail(self):
        run = self.make_run()
        tensor = self.out_tensor([0.0, 0.0])

        def explode(session):
            raise ValueError('reference failed')

        with self.assertRaises(ro.Row1OracleError) as caught:
            self.output(run, explode, tensor)
        self.assertIn('oracle_or_candidate_exception', str(caught.exception))
        self.assertEqual(self.store.failures()[0].kind,
                         'oracle_or_candidate_exception')
        run = self.make_run()
        tensor = self.out_tensor([float('inf'), 0.0])

        def fine(session):
            session.route('sparse_uva', 1)

        with self.assertRaises(ro.Row1OracleError) as caught:
            self.output(run, fine, tensor)
        self.assertIn('nonfinite_qsa_output', str(caught.exception))

    def test_release_requires_the_output_comparison_or_a_recorded_cancel(self):
        # A layer with the original bound but no comparison yet may not release.
        run = self.make_run(row1_reference=lambda: None)
        tensor = self.out_tensor([0.0])
        with self.assertRaises(et.E3Pending):
            self.qualifier.check_releasable(run)

        def reference(session):
            session.route('sparse_uva', 1)

        self.output(run, reference, tensor)
        self.assertTrue(self.qualifier.check_releasable(run))
        cancelled = self.make_run(layer_name='qsa.1')
        self.assertTrue(self.qualifier.check_releasable(cancelled,
                                                        cancelled=True))
        self.assertEqual(self.qualifier.stats['aborted'], 1)
        absent = self.make_run(layer_name='qsa.2')
        self.assertTrue(self.qualifier.check_releasable(absent))
        held = self.make_run(layer_name='qsa.3',
                        sparse_hold=types.SimpleNamespace(released=False))
        with self.assertRaises(ro.Row1OracleError) as caught:
            self.qualifier.check_releasable(held)
        self.assertIn('no row1 reference', str(caught.exception))

    def test_finish_layer_publishes_the_pass_and_hold_release(self):
        run = self.make_run(sparse_hold=types.SimpleNamespace(released=False))
        bucket = self.bucket(segments=((12, 0, 5),))
        self.fill_actual(bucket)
        self.qualifier.bucket(run, bucket, self.arena)
        tensor = self.out_tensor([0.0, 0.0])

        def reference(session):
            session.route('sparse_uva', 2)

        self.output(run, reference, tensor)
        run.sparse_hold.released = True
        record = self.qualifier.finish_layer(
            run, hold_stats={'buckets': 1, 'consumed': 1})
        self.assertTrue(record.passed)
        self.assertTrue(record.hold_released)
        self.assertEqual(record.route, 'sparse_dma')
        self.assertEqual(record.bucket_checks, 1)
        self.assertEqual(record.bucket_rows, 5)
        self.assertEqual(record.output_checks, 1)
        self.assertEqual(record.output_rows, 2)
        self.assertEqual(dict(record.hold_stats), {'buckets': 1, 'consumed': 1})
        self.assertEqual(len(self.store.layer_records()), 1)
        self.assertEqual(self.qualifier.stats['layers'], 1)

    def test_finish_layer_without_a_pass_is_refused(self):
        run = self.make_run(sparse_hold=types.SimpleNamespace(released=True))
        with self.assertRaises(ro.Row1OracleError) as caught:
            self.qualifier.finish_layer(run, hold_stats={})
        self.assertIn('without a passing comparison', str(caught.exception))


class Row1ScopeTests(unittest.TestCase):
    def setUp(self):
        from test_e3_native_integration import make_worker_stack
        self.stack = make_worker_stack(0)
        self.server = self.stack.server
        self.providers = []
        def provider(store, epoch, generation):
            self.providers.append((store, epoch, generation))
            return (None if store is None else lambda operation: store), ('qsa.0',)
        self.server.row1_provider = provider

    def configure(self, enabled=True, run_id='native-row1'):
        return self.server.configure_row1(enabled=enabled, run_id=run_id,
                                          engine_epoch=1, route_generation=3)

    def test_fresh_scope_enable_disable_retains_exact_raw_records(self):
        self.assertFalse(self.server.query_row1()['enabled'])
        first = self.configure()
        self.assertTrue(first['enabled'])
        store = self.server.row1_store
        record = Row1RecordStoreTests().record()
        store.append_comparison(record)
        self.assertIs(self.server.row1_factory(None), store)
        stopped = self.configure(False)
        self.assertFalse(stopped['enabled'])
        self.assertFalse(stopped['raw']['enabled'])
        self.assertIsNone(self.server.row1_factory)
        self.assertIs(self.server.row1_store, store)
        self.assertEqual(stopped['raw']['records'], (record.as_dict(),))
        self.assertEqual(stopped['schema'], 's5-native-row1-store/1')
        self.assertEqual(stopped['local_layers'], ['qsa.0'])
        self.assertEqual(stopped['worker_incarnation'], self.stack.state.incarnation)
        # An idempotent disable and a later fresh scope do not erase evidence.
        self.assertEqual(self.configure(False), stopped)
        self.configure(run_id='next-row1')
        self.assertIsNot(self.server.row1_store, store)
        self.assertEqual(self.server.row1_evidence(run_id='native-row1'), stopped)

    def test_scopes_cannot_reopen_or_replace_enabled_scope(self):
        self.configure()
        with self.assertRaisesRegex(Exception, 'still enabled'):
            self.configure(run_id='foreign')
        self.configure(False)
        with self.assertRaisesRegex(Exception, 'cannot be reopened'):
            self.configure()
        with self.assertRaisesRegex(Exception, 'unknown scope'):
            self.configure(False, run_id='missing')

    def test_live_prefill_or_foreign_epoch_refuses_before_provider(self):
        self.stack.state.prefill.handle = object()
        with self.assertRaises(et.E3Pending):
            self.configure()
        self.assertEqual(self.providers, [])
        self.stack.state.prefill.handle = None
        with self.assertRaisesRegex(Exception, 'stale engine epoch'):
            self.server.configure_row1(enabled=True, run_id='row1',
                                        engine_epoch=2, route_generation=3)
        self.assertEqual(self.providers, [])

    def test_provider_failure_keeps_old_store_and_configuration(self):
        self.configure()
        store = self.server.row1_store
        factory = self.server.row1_factory
        def failed(*args):
            raise RuntimeError('actual drain failed')
        self.server.row1_provider = failed
        with self.assertRaisesRegex(RuntimeError, 'actual drain failed'):
            self.configure(False)
        self.assertIs(self.server.row1_store, store)
        self.assertIs(self.server.row1_factory, factory)
        self.assertTrue(self.server.row1_evidence(run_id='native-row1')['enabled'])

    def test_retained_request_refuses_even_after_transaction_release(self):
        self.server._flash_s5_bridge = types.SimpleNamespace(
            active=False, _retained=object(), _source_binding=None,
            _source_close=None)
        with self.assertRaises(et.E3Pending):
            self.configure()
        self.assertEqual(self.providers, [])

    def test_raw_enabled_reports_the_factory_not_store_presence(self):
        self.configure()
        self.server.row1_factory = None
        self.assertFalse(self.server.query_row1()['enabled'])


class Row1RecordStoreTests(unittest.TestCase):
    """Bounded, immutable, queryable records with no clear/drop."""

    def record(self, layer='qsa.0', bucket=0):
        return ro.Row1Comparison(
            identity=ro.Row1Identity(0, 'worker', 1, 0, 1, 0, 1, 1, layer,
                                     'sparse_row_bucket'),
            bucket=bucket, route='sparse_dma', rows=1, pieces=1,
            compact_start=0, compared_bytes=1, different_bytes=None,
            equal=True, finite=None, coverage_rows=1, coverage_bytes=1,
            observed_ns=0, digest='0' * 64)

    def test_capacity_is_a_refusal_not_a_silent_drop(self):
        store = ro.Row1RecordStore(capacity=1)
        store.append_comparison(self.record())
        with self.assertRaises(ro.Row1OracleError) as caught:
            store.append_comparison(self.record(bucket=1))
        self.assertIn('store is full', str(caught.exception))
        self.assertEqual(len(store.records()), 1)
        self.assertFalse(hasattr(store, 'clear'))
        self.assertFalse(hasattr(store, 'drop'))

    def test_queries_filter_by_layer_and_bound_the_count(self):
        store = ro.Row1RecordStore()
        for layer in ('qsa.0', 'qsa.1', 'qsa.0'):
            store.append_comparison(self.record(layer=layer))
        self.assertEqual(len(store.records()), 3)
        self.assertEqual(len(store.records(layer='qsa.0')), 2)
        self.assertEqual(len(store.records(limit=1)), 1)
        self.assertEqual(len(store.records(layer='qsa.0', limit=1)), 1)
        with self.assertRaises(ro.Row1OracleError):
            store.records(layer='')
        with self.assertRaises(ro.Row1OracleError):
            store.records(limit=-1)
        summary = store.summary()
        self.assertEqual(summary['comparisons'], 3)
        self.assertIn('not shipping performance', summary['scope'])

    def test_wrong_record_types_are_refused(self):
        store = ro.Row1RecordStore()
        with self.assertRaises(ro.Row1OracleError):
            store.append_comparison(object())
        with self.assertRaises(ro.Row1OracleError):
            store.append_layer(object())
        with self.assertRaises(ro.Row1OracleError):
            store.append_failure(object())


# --- generated helper: the actual oracle over the actual arithmetic -----------

class GeneratedRow1Tests(sparse_test._GeneratedHarness, unittest.TestCase):
    """The generated sparse helper drives the E2 oracle end to end (fake bytes).

    The attention kernel is the CPU fixture (it writes no output); the pinned
    original ``flash_e2_arena_attention`` is replaced by a stub that writes the
    same baseline output bytes and declares its route, exactly as the original
    would after its own UVA gather.  The arena bytes are the real compact-row
    copy emitted by the operation; the comparison is the actual E2 mechanism.
    """

    def setUp(self):
        self.reference_route = 'sparse_uva'
        self.reference_pages = 15
        self.reference_output = b'OUT'
        self.reference_bad_row = None
        self.store = ro.Row1RecordStore(capacity=64)
        super().setUp()

    def reference(self, q, kv, logical, table, rows, max_page, arena,
                  use_prefill_config, out, *, e2):
        e2.route(self.reference_route, self.reference_pages)
        bad = self.reference_bad_row
        for index in range(int(out.shape[0])):
            out.storage.rows[out.row_offset + index] = (
                b'BAD' if bad == index else self.reference_output)

    def compile_helper(self):
        package_name = 'models.qwen4_exp.nvidia.ops.qsa'
        package = sparse_test._module(
            package_name,
            flash_s5_sparse_arena_attention=self.attention,
            flash_s5_arena_attention=self.attention,
            flash_e2_arena_attention=self.reference,
            qsa_get_staging_arena=lambda kv_cache, nbytes: self.arena)
        for name in ('models', 'models.qwen4_exp', 'models.qwen4_exp.nvidia',
                     'models.qwen4_exp.nvidia.ops'):
            sys.modules.setdefault(name, sparse_test._module(name))
        sys.modules[package_name] = package
        self.addCleanup(sys.modules.pop, package_name, None)
        source = sparse_test.ast.Module(
            body=[sparse_test._function_node(
                self.generated_qsa, '_flash_s5_sparse_staged_operation')],
            type_ignores=[])
        sparse_test.ast.fix_missing_locations(source)
        namespace = {'__name__': 'models.qwen4_exp.nvidia.qsa',
                     '__package__': 'models.qwen4_exp.nvidia',
                     'torch': self.torch}
        exec(compile(source, '<generated sparse qsa>', 'exec'), namespace)
        return namespace['_flash_s5_sparse_staged_operation']

    def make_operation(self, layer_tables):
        lane = sparse_test.make_lane()
        transaction = sparse_test.make_transaction(chunk_end=1 << 24)
        operation = qo.QsaPrefillOperation(
            transaction, lane, device=self.device,
            plan_provider=qo.metadata_plan_provider(
                sparse_test.Snapshot([sparse_test.Row()], range(260)),
                layer_tables=tuple(layer_tables)))
        operation.state_epoch = 1
        return operation, lane, transaction

    def case(self, *, backend=None, output_pages=25):
        """Register one layer, run the generated helper through the oracle."""
        tokens = 24 * PAGE_ROWS + 17
        mapping = tuple(range(output_pages))
        operation, lane, transaction = self.make_operation(
            [('qsa.0', mapping)])
        pool = max(max(mapping) + 1, (tokens + PAGE_ROWS - 1) // PAGE_ROWS)
        source = self.make_source(pool)
        versions = tuple(range(1, pool + 3))
        operation.register_source('qsa.0', source, versions=versions)
        operation.begin_forward(layers=['qsa.0'])
        self.device.auto_ready = False
        operation.begin_layer('qsa.0', source)
        operation.producer_submitted('qsa.0', source)
        hold = shd.SparseSourceHold(
            sparse_test.make_owner(lane, 'qsa.0', mapping, tokens,
                                   transaction=transaction),
            allocator=sparse_test.FakeAllocator())
        operation.attach_sparse_source('qsa.0', hold)
        operation.qualification = ro.Row1Qualifier(
            self.store, transaction=transaction, lane=lane,
            backend=backend or RowBackend(),
            epochs=lambda: dict(worker_incarnation='generated-worker',
                                engine_epoch=2, route_generation=1))
        layer = types.SimpleNamespace(
            layer_name='qsa.0', _qsa_kv_offload=True, _qsa_stage_min_rows=1,
            _qsa_stage_arena_bytes=ARENA_ROWS * ROW_BYTES)
        width = 2
        flat = []
        for row in range(tokens):
            flat.append(row % 60000)
            flat.append(-1)
        logical = T((tokens, width), device='cuda:0', dtype='torch.int32',
                    values=flat)
        query = T((tokens, QUERY_HEADS, HEAD_DIM), device='cuda:0',
                  dtype='torch.bfloat16')
        out_rows = Rows(tokens)
        out_rows.rows = [b'OUT'] * tokens
        output = T((tokens, QUERY_HEADS, HEAD_DIM), device='cuda:0',
                   dtype='torch.bfloat16', storage=out_rows)
        metadata = types.SimpleNamespace(
            _qsa_host_meta=((0, tokens), (tokens,)),
            query_start_loc=T((2,), device='cpu', dtype='torch.int32'),
            seq_lens=T((1,), device='cpu', dtype='torch.int32'),
            block_table=T((1, 260), device='cuda:0', dtype='torch.int64',
                          values=[0] * 260))
        generator = self.helper(operation, layer, query, source, logical,
                                metadata, tokens, True, output)
        operation.attach_continuation(generator)
        return operation, hold, tokens

    def drive(self, operation):
        sentinels = 0
        buckets = 0
        while True:
            try:
                value = operation.advance()
            except StopIteration:
                return sentinels, buckets
            if getattr(value, 'sparse_selection_pending', False):
                sentinels += 1
                self.device.ready_all()
                continue
            buckets += 1
            self.device.ready_all()

    def test_generated_helper_qualifies_bucket_and_output_bytes(self):
        operation, hold, tokens = self.case()
        sentinels, buckets = self.drive(operation)
        self.assertEqual(sentinels, 1)
        self.assertEqual(buckets, 1)
        run = operation._layers['qsa.0']
        self.assertEqual(run.sparse.route, 'sparse_dma')
        self.assertIsNone(run.qualification_bucket)
        self.assertIsNone(run.row1_failed)
        self.assertIs(run.uva, run.source)
        records = self.store.records()
        self.assertEqual(len(records), 2)
        bucket, output = records
        self.assertEqual(bucket.identity.operation, 'sparse_row_bucket')
        self.assertEqual(bucket.route, 'sparse_dma')
        self.assertTrue(bucket.equal)
        self.assertEqual(bucket.rows, run.sparse.plan.bucket_rows[0])
        self.assertEqual(output.identity.operation, 'qsa_output_before_moe')
        self.assertEqual(output.route, 'sparse_dma')
        self.assertTrue(output.equal)
        self.assertTrue(output.finite)
        self.assertEqual(output.rows, tokens)
        self.assertEqual(output.coverage_rows, tokens)
        self.assertEqual(output.identity.engine_epoch, 2)
        self.assertEqual(output.identity.route_generation, 1)
        self.assertEqual(output.identity.rank, 0)
        self.assertEqual(output.identity.state_epoch, 1)
        self.assertEqual(output.identity.source_epoch, run.source_epoch)
        self.assertEqual(output.identity.mapping_version,
                         run.mapping_version)
        self.assertEqual(output.identity.worker_incarnation,
                         'generated-worker')
        # The original wrote the same bytes; both snapshots are the output
        # before MoE, not a digest of it.
        self.assertEqual(output.different_bytes, None)
        operation.end_layer('qsa.0')
        operation.end_forward()
        self.device.ready_all()
        self.assertTrue(operation.drain())
        operation.release()
        self.assertTrue(hold.released)
        # The bounded raw store survives release and stays queryable.
        self.assertEqual(self.store.records(layer='qsa.0'), tuple(records))
        self.assertEqual(self.store.failures(layer='qsa.0'), ())
        layers = self.store.layer_records()
        self.assertEqual(len(layers), 1)
        self.assertTrue(layers[0].passed)
        self.assertTrue(layers[0].hold_released)
        self.assertEqual(layers[0].route, 'sparse_dma')
        self.assertEqual(layers[0].bucket_checks, 1)
        self.assertEqual(layers[0].output_checks, 1)
        self.assertEqual(len(layers[0].digest), 64)
        inventory = operation.inventory()
        self.assertTrue(inventory['row1_enabled'])
        self.assertEqual(inventory['row1_buckets'], 1)
        self.assertEqual(inventory['row1_outputs'], 1)
        self.assertEqual(inventory['row1_layers'], 1)
        self.assertEqual(inventory['sparse_source_released'], 1)

    def test_a_mutated_bucket_means_the_hold_is_never_released(self):
        operation, hold, _tokens = self.case(
            backend=RowBackend(gather_mutation=12))
        with self.assertRaises(ro.Row1OracleError) as caught:
            self.drive(operation)
        self.assertIn('arena_bytes', str(caught.exception))
        run = operation._layers['qsa.0']
        self.assertEqual(run.row1_failed, 'arena_bytes')
        failures = self.store.failures()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].kind, 'arena_bytes')
        self.assertGreaterEqual(failures[0].different_bytes, 1)
        self.assertFalse(failures[0].hold_released)
        self.assertEqual(len(self.store.records()), 1)
        self.assertFalse(self.store.records()[0].equal)
        self.assertFalse(hold.released)
        self.assertTrue(operation._failed)
        operation.cancel('row1 mutation fixture')
        with self.assertRaises(et.E3Pending):
            operation.release()
        self.assertFalse(hold.released)
        # The raw record store survives and stays queryable after the failure.
        self.assertEqual(len(self.store.failures(layer='qsa.0')), 1)

    def test_a_mutated_output_byte_fails_the_qsa_oracle(self):
        self.reference_bad_row = 17
        operation, hold, _tokens = self.case()
        with self.assertRaises(ro.Row1OracleError) as caught:
            self.drive(operation)
        self.assertIn('qsa_output_bytes', str(caught.exception))
        run = operation._layers['qsa.0']
        self.assertEqual(run.row1_failed, 'qsa_output_bytes')
        failure = self.store.failures()[0]
        self.assertEqual(failure.kind, 'qsa_output_bytes')
        self.assertEqual(failure.different_bytes, 1)
        records = self.store.records()
        self.assertEqual(len(records), 2)
        self.assertTrue(records[0].equal)
        self.assertFalse(records[1].equal)
        self.assertFalse(hold.released)

    def test_provenance_pins_the_row1_calls_and_the_driver_source(self):
        # Both package manifests ship the new runtime modules; the generated
        # package imports them in its closure test.
        import decode_probe
        import make_s5_overlap
        for name in ('native_source', 'row1_oracle'):
            self.assertIn(name, decode_probe.GENERATED_SOURCES)
            self.assertIn(name, make_s5_overlap.E3_MODULES)
            self.assertNotIn(name, decode_probe.BUILD_ONLY_SOURCES)
        helper = operation_source.SPARSE_HELPER
        self.assertEqual(helper.count('operation.bind_row1_uva('), 1)
        self.assertEqual(helper.count('operation.bind_row1_reference('), 1)
        self.assertEqual(helper.count('operation.row1_output('), 1)
        self.assertEqual(helper.count(
            'from .ops.qsa import flash_e2_arena_attention'), 1)
        self.assertEqual(operation_source.revert_layer(self.generated_qsa),
                         self.qsa_e2)
        # The generated sparse helper is the D2-composed QSA source the layer
        # driver pins; the E2 bytes are its revert target.
        digest = hashlib.sha256(self.generated_qsa.encode()).hexdigest()
        self.assertEqual(
            qsa_driver.PINS['models/qwen4_exp/nvidia/qsa.py'], digest)
        generated = qsa_driver.generate_one(
            'models/qwen4_exp/nvidia/qsa.py', self.generated_qsa)
        self.assertEqual(qsa_driver.revert_one(
            'models/qwen4_exp/nvidia/qsa.py', generated,
            expected=self.generated_qsa), self.generated_qsa)


if __name__ == '__main__':
    unittest.main()
