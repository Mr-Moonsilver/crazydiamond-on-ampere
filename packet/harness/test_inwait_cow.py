#!/usr/bin/env python3
"""inwait build-4 (results/hub/inwait/DESIGN-4.md): the fast CoW copy against stock's bytes, the zero/copy row, the reserved cap. Discovery mode.

The copy runs on real CPU tensors against the image's real ``copy_kv_cache_blocks_inplace``, in both of its branches. No GPU: whether it is
faster, and what the residue chunk's device time is on each TP rank, is the box's question.
"""
import itertools
import logging
import os
import re
import types
import unittest
from unittest import mock

import torch
from vllm.utils import torch_utils
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

import lane_chunk as lc
import worker_bridge as wb

NUM_BLOCKS = 8
torch_utils.PIN_MEMORY = False    # no driver on the hub: stock pins its index tensor, which needs one


def whole_storage(dtype=torch.int16):
    # The cache covers its whole storage: stock's byte view branch.
    return torch.arange(NUM_BLOCKS * 3 * 5, dtype=dtype).view(NUM_BLOCKS, 3, 5).clone()


def split_blocks():
    # Virtual block splitting: 2 kernel blocks a scheduler block.
    return torch.arange(NUM_BLOCKS * 2 * 4, dtype=torch.float32).view(NUM_BLOCKS * 2, 4).clone()


def strided():
    # A view into a wider allocation: stock's unflatten branch.
    base = torch.arange(NUM_BLOCKS * 6 * 3, dtype=torch.int32).view(NUM_BLOCKS, 6, 3).clone()
    return base[:, ::2, :], base


def caches():
    shared = whole_storage()
    wide = torch.arange(NUM_BLOCKS * 7, dtype=torch.int64).view(NUM_BLOCKS, 7).clone()
    view, _base = strided()
    # Aliased views of one allocation: cross-layer sharing.
    alias = wide.view(NUM_BLOCKS, 7)
    return [shared, split_blocks(), view, wide, alias]


def bytes_of(items):
    return [item.clone().numpy().tobytes() for item in items]


class Copy(unittest.TestCase):
    def assert_same_as_stock(self, pairs):
        stock, fast = caches(), caches()
        copy_kv_cache_blocks_inplace(stock, NUM_BLOCKS, [KVCacheBlockCopy(s, d) for s, d in pairs])
        out = lc.copy_blocks_fast(fast, NUM_BLOCKS, pairs)
        self.assertIsNotNone(out)
        self.assertEqual(bytes_of(fast), bytes_of(stock))
        return out

    def test_every_single_pair_copies_the_same_bytes(self):
        for src, dst in itertools.permutations(range(NUM_BLOCKS), 2):
            self.assert_same_as_stock([(src, dst)])

    def test_several_pairs_copy_the_same_bytes(self):
        self.assert_same_as_stock([(0, 3), (1, 4), (2, 5)])
        self.assert_same_as_stock([(0, 3), (0, 4)])

    def test_the_rest_of_the_cache_is_untouched(self):
        before, after = caches(), caches()
        lc.copy_blocks_fast(after, NUM_BLOCKS, [(1, 6)])
        for old, new in zip(before, after):
            kernel = old.shape[0] // NUM_BLOCKS
            old, new = old.unflatten(0, (NUM_BLOCKS, kernel)), new.unflatten(0, (NUM_BLOCKS, kernel))
            for block in range(NUM_BLOCKS):
                want = old[1] if block == 6 else old[block]
                self.assertEqual(new[block].numpy().tobytes(), want.numpy().tobytes())

    def test_an_aliased_view_is_copied_once(self):
        wide = torch.arange(NUM_BLOCKS * 7, dtype=torch.int64).view(NUM_BLOCKS, 7).clone()
        out = lc.copy_blocks_fast([wide, wide.view(NUM_BLOCKS, 7)], NUM_BLOCKS, [(1, 2)])
        self.assertEqual(out['views'], 1)
        self.assertEqual(out['bytes'], 7 * 8)

    def test_a_list_where_order_could_matter_goes_to_stock_with_nothing_copied(self):
        for pairs in ([(0, 1), (1, 2)], [(0, 2), (1, 2)], [(3, 3)]):
            items = caches()
            before = bytes_of(items)
            self.assertIsNone(lc.copy_blocks_fast(items, NUM_BLOCKS, pairs), pairs)
            self.assertEqual(bytes_of(items), before)

    def test_blocks_not_divisible_are_refused(self):
        with self.assertRaises(lc.LaneChunkError):
            lc.copy_blocks_fast([torch.zeros(NUM_BLOCKS + 1, 2)], NUM_BLOCKS, [(0, 1)])


class Mark:
    # one timing event
    def __init__(self, log=None, done=True, at=0.0):
        self.done, self.at, self.log = done, at, log

    def query(self):
        return self.done

    def elapsed_time(self, other):
        return other.at - self.at


class Pages(unittest.TestCase):
    def setUp(self):
        saved = dict(lc.COW_FAST)
        self.addCleanup(lambda: (lc.COW_FAST.update(saved), lc.take_cow_row()))
        lc.COW_FAST.update(selected=False, calls=0, pairs=0, fallbacks=0, rows=0, row_errors=0)
        lc.take_cow_row()
        self.log = []
        self.caches = caches()
        zeroer = types.SimpleNamespace(zero_block_ids=lambda ids: self.log.append(('zero', tuple(ids))))
        self.runner = types.SimpleNamespace(kv_caches=self.caches, kv_cache_config=types.SimpleNamespace(num_blocks=NUM_BLOCKS), kv_block_zeroer=zeroer)
        marks = [Mark(at=0.0), Mark(at=1.5), Mark(at=501.5), Mark(at=503.5)]
        self.ports = types.SimpleNamespace(block_copy=KVCacheBlockCopy, copy_blocks_inplace=self.stock, cow_mark=lambda: marks.pop(0))
        self.request = lc.LaneRequest.__new__(lc.LaneRequest)
        self.request.runner, self.request.ports = self.runner, self.ports

    def stock(self, kv_caches, num_blocks, copies):
        self.log.append(('stock', len(copies)))
        copy_kv_cache_blocks_inplace(kv_caches, num_blocks, copies)

    def admission(self, zero=(), pairs=()):
        return types.SimpleNamespace(new_block_ids_to_zero=zero, cow_pairs=pairs)

    def test_off_runs_stock_and_times_it(self):
        self.request._apply_allocated_pages(self.admission(zero=(5,), pairs=((1, 2),)))
        self.assertEqual(self.log, [('zero', (5,)), ('stock', 1)])
        self.assertEqual((lc.COW_FAST['calls'], lc.COW_FAST['rows']), (0, 1))
        row = lc.cow_row_times(lc.take_cow_row())
        self.assertEqual((row['zero_ids'], row['pairs'], row['fast'], row['views'], row['bytes']), (1, 1, 0, None, None))
        self.assertEqual((row['zero_dev_us'], row['copy_dev_us'], row['host_view_dev_us']), (1500, 500000, 2000))
        self.assertNotIn('marks', row)
        self.assertIsNone(lc.take_cow_row(), 'once')

    def test_on_runs_the_fast_copy_with_the_same_bytes(self):
        stock = caches()
        copy_kv_cache_blocks_inplace(stock, NUM_BLOCKS, [KVCacheBlockCopy(1, 2)])
        lc.select_cow_fast(True)
        self.request._apply_allocated_pages(self.admission(pairs=((1, 2),)))
        self.assertEqual(self.log, [])
        self.assertEqual(bytes_of(self.caches), bytes_of(stock))
        self.assertEqual((lc.COW_FAST['calls'], lc.COW_FAST['pairs'], lc.COW_FAST['fallbacks']), (1, 1, 0))
        row = lc.take_cow_row()
        self.assertEqual(row['fast'], 1)
        self.assertGreater(row['bytes'], 0)

    def test_on_with_a_chain_goes_to_stock_and_counts_it(self):
        lc.select_cow_fast(True)
        self.request._apply_allocated_pages(self.admission(pairs=((1, 2), (2, 3))))
        self.assertEqual(self.log, [('stock', 2)])
        self.assertEqual((lc.COW_FAST['calls'], lc.COW_FAST['fallbacks']), (0, 1))
        self.assertEqual(lc.take_cow_row()['fast'], 0)

    def test_host_views_copy_the_same_blocks(self):
        views = [torch.arange(NUM_BLOCKS * 2, dtype=torch.int32).view(NUM_BLOCKS, 2).clone()]
        self.runner._qsa_host_views = views
        self.request._apply_allocated_pages(self.admission(pairs=((1, 2),)))
        self.assertEqual(views[0][2].tolist(), views[0][1].tolist())
        row = lc.take_cow_row()
        self.assertEqual((row['host_views'], row['host_fast'], row['host_bytes']), (1, 0, None))

    def test_on_copies_the_host_views_fast_too_with_the_same_bytes(self):
        views = [torch.arange(NUM_BLOCKS * 2, dtype=torch.int32).view(NUM_BLOCKS, 2).clone()]
        want = views[0].clone()
        want[2] = want[1]
        self.runner._qsa_host_views = views
        lc.select_cow_fast(True)
        self.request._apply_allocated_pages(self.admission(pairs=((1, 2),)))
        self.assertEqual(self.log, [])
        self.assertEqual(views[0].tolist(), want.tolist())
        row = lc.take_cow_row()
        self.assertEqual((row['host_fast'], row['host_bytes']), (1, 2 * 4))

    def test_on_with_a_chain_copies_the_host_views_with_stock_too(self):
        views = [torch.arange(NUM_BLOCKS * 2, dtype=torch.int32).view(NUM_BLOCKS, 2).clone()]
        self.runner._qsa_host_views = views
        lc.select_cow_fast(True)
        self.request._apply_allocated_pages(self.admission(pairs=((1, 2), (2, 3))))
        self.assertEqual(self.log, [('stock', 2), ('stock', 2)])
        self.assertEqual(lc.take_cow_row()['host_fast'], 0)

    def filtered(self, views, tables, pairs, **runner):
        self.runner._qsa_host_views = views
        self.runner.__dict__.update(runner)
        lc.select_cow_host_filter(True)
        self.addCleanup(lc.select_cow_host_filter, False)
        self.request._apply_allocated_pages(types.SimpleNamespace(new_block_ids_to_zero=(), cow_pairs=pairs, block_tables=tables))
        return lc.take_cow_row()

    def view(self):
        return torch.arange(NUM_BLOCKS * 2, dtype=torch.int32).view(NUM_BLOCKS, 2).clone()

    def test_filter_on_a_target_outside_the_qsa_groups_copies_no_host_page(self):
        lc.COW_HOST_FILTER.update(kept=0, skipped=0, unfiltered=0)
        views = [self.view()]
        before = views[0].clone()
        row = self.filtered(views, ((7,), (2,)), ((1, 2),), _qsa_host_groups=[0])
        self.assertEqual(views[0].tolist(), before.tolist())
        self.assertEqual(self.log, [('stock', 1)], 'the device pass still ran')
        self.assertEqual((row['host_pairs'], row['pages']['pairs'], row['pages']['host_pairs']), (0, [(1, 2)], []))
        self.assertEqual((lc.COW_HOST_FILTER['kept'], lc.COW_HOST_FILTER['skipped']), (0, 1))

    def test_filter_on_a_consumer_hit_in_a_qsa_group_is_still_copied(self):
        lc.COW_HOST_FILTER.update(kept=0, skipped=0, unfiltered=0)
        views = [self.view()]
        want = views[0].clone()
        want[2] = want[1]
        row = self.filtered(views, ((2, 3), (6,)), ((1, 2), (4, 6)), _qsa_host_groups=[0])
        self.assertEqual(views[0].tolist(), want.tolist(), 'pair (4, 6) is not in the QSA group')
        self.assertEqual(row['pages']['host_pairs'], [(1, 2)])
        self.assertEqual((lc.COW_HOST_FILTER['kept'], lc.COW_HOST_FILTER['skipped']), (1, 1))

    def test_filter_with_the_groups_unknown_copies_everything_and_counts_it(self):
        lc.COW_HOST_FILTER.update(kept=0, skipped=0, unfiltered=0)
        views = [self.view()]
        want = views[0].clone()
        want[2] = want[1]
        self.filtered(views, ((7,), (2,)), ((1, 2),))
        self.assertEqual(views[0].tolist(), want.tolist())
        self.assertEqual(lc.COW_HOST_FILTER['unfiltered'], 1)

    def test_filter_and_fast_together_copy_the_kept_pair_fast(self):
        views = [self.view()]
        want = views[0].clone()
        want[2] = want[1]
        lc.select_cow_fast(True)
        row = self.filtered(views, ((2,), (6,)), ((1, 2), (4, 6)), _qsa_host_groups=[0])
        self.assertEqual(views[0].tolist(), want.tolist())
        self.assertEqual((row['fast'], row['host_fast'], row['host_pairs'], row['host_bytes']), (1, 1, 1, 8))

    def test_filter_off_copies_every_pair_as_before(self):
        views = [self.view()]
        want = views[0].clone()
        want[2] = want[1]
        self.runner._qsa_host_views = views
        self.runner._qsa_host_groups = [0]
        self.request._apply_allocated_pages(types.SimpleNamespace(new_block_ids_to_zero=(), cow_pairs=((1, 2),), block_tables=((7,),)))
        self.assertEqual(views[0].tolist(), want.tolist())

    def test_a_chunk_with_no_pages_has_no_row_and_no_mark(self):
        self.ports.cow_mark = None
        self.request._apply_allocated_pages(self.admission())
        self.assertIsNone(lc.take_cow_row())
        self.assertEqual(self.log, [])

    def test_a_port_without_marks_still_copies(self):
        self.ports.cow_mark = None
        self.request._apply_allocated_pages(self.admission(pairs=((1, 2),)))
        row = lc.cow_row_times(lc.take_cow_row())
        self.assertEqual((row['zero_dev_us'], row['copy_dev_us'], row['host_view_dev_us']), (None, None, None))

    def test_an_unfinished_pair_reads_none_and_never_waits(self):
        row = dict(marks=[Mark(), Mark(), Mark(done=False), Mark()])
        out = lc.cow_row_times(row)
        self.assertEqual((out['zero_dev_us'], out['copy_dev_us'], out['host_view_dev_us']), (None, None, None))


PAGE_TOKENS, HOST_WIDTH = 8, 4


class Residency(unittest.TestCase):
    """Two QSA pools with their UVA views and placeholders, one other KV allocation: the box's arenas as far as CPU tensors can stand in."""

    def setUp(self):
        shape = (NUM_BLOCKS, 1, PAGE_TOKENS, HOST_WIDTH)
        self.hosts = [torch.randn(shape).to(torch.bfloat16) for _ in range(2)]
        self.slots = [torch.zeros(NUM_BLOCKS, 1, PAGE_TOKENS, 1, dtype=torch.bfloat16) for _ in range(2)]
        self.other = whole_storage()
        ctx = {'qsa%d' % i: types.SimpleNamespace(_qsa_host_kv=h, _qsa_gpu_slots=s) for i, (h, s) in enumerate(zip(self.hosts, self.slots))}
        ctx['plain'] = types.SimpleNamespace()
        self.views = [h.view(shape) for h in self.hosts]
        self.runner = types.SimpleNamespace(kv_caches=self.slots + [self.other], kv_cache_config=types.SimpleNamespace(num_blocks=NUM_BLOCKS),
                                            compilation_config=types.SimpleNamespace(static_forward_context=ctx))

    def test_each_pass_counts_each_pair_by_the_arena_it_sits_in(self):
        out = lc.cow_residency(self.runner, self.views, [(1, 2), (3, 4)])
        page = PAGE_TOKENS * HOST_WIDTH * 2
        self.assertEqual(out, {'k_slot_slot_pairs': 4, 'k_slot_slot_bytes': 4 * PAGE_TOKENS * 2,
                               'k_device_device_pairs': 2, 'k_device_device_bytes': 2 * 3 * 5 * 2,
                               'h_host_host_pairs': 4, 'h_host_host_bytes': 4 * page})

    def test_a_view_outside_every_arena_reads_other(self):
        out = lc.cow_residency(self.runner, [torch.zeros(NUM_BLOCKS, 2)], [(1, 2)])
        self.assertEqual(out['h_other_other_pairs'], 1)

    def test_no_views_counts_only_the_first_pass(self):
        self.assertFalse([k for k in lc.cow_residency(self.runner, None, [(1, 2)]) if k.startswith('h_')])

    def copied(self, chunk_start=PAGE_TOKENS * 3 + 5):
        for host in self.hosts:
            host[2] = host[1]
        return lc.cow_host_verify(self.runner, [(1, 2)], chunk_start)

    def test_a_copied_prefix_verifies_and_the_rest_of_the_page_may_differ(self):
        for host in self.hosts:
            host[2, :, 5:] = 7.0
        for host in self.hosts:
            host[2, :, :5] = host[1, :, :5]
        self.assertEqual(lc.cow_host_verify(self.runner, [(1, 2)], PAGE_TOKENS * 3 + 5), dict(verify_pairs=2, verify_unequal=0, verify_prefix=5))

    def test_a_nan_equals_itself(self):
        self.hosts[0][1, 0, 0, 0] = float('nan')
        self.assertEqual(self.copied()['verify_unequal'], 0)

    def test_a_wrong_or_missing_copy_is_counted_per_pool(self):
        self.copied()
        self.hosts[1][2, 0, 4, 0] = self.hosts[1][1, 0, 4, 0] + 1
        self.assertEqual(lc.cow_host_verify(self.runner, [(1, 2)], PAGE_TOKENS * 3 + 5)['verify_unequal'], 1)

    def test_the_line_carries_counts_and_no_pointer(self):
        lc._COW_ROW[0] = dict(marks=[Mark(), Mark(), Mark(), Mark()], zero_ids=0, pairs=1, fast=0, views=None, bytes=None, host_views=2,
                              host_fast=0, host_bytes=None, zero_host_us=1, copy_host_us=2, host_view_us=3,
                              pages=dict(runner=self.runner, views=self.views, pairs=[(1, 2)]))
        lines = []
        with mock.patch.dict(os.environ, {'FLASH_S5_INWAIT': '1'}), \
                mock.patch('native_measurement.emit_receipt_kv', lambda line, logger: lines.append(line)):
            wb.NativeWorkerBridge._cow_receipt(types.SimpleNamespace(rank=0), types.SimpleNamespace(txn_seq=1, chunk_start=PAGE_TOKENS + 5, chunk_end=99))
        self.assertEqual(len(lines), 1)
        self.assertIn('h_host_host_pairs=2', lines[0])
        self.assertIn('verify_pairs=2', lines[0])
        self.assertNotIn('pages=', lines[0])
        self.assertNotIn('namespace', lines[0])
        pointers = {str(t.data_ptr()) for t in self.hosts + self.slots + [self.other]}
        pointers |= {hex(t.data_ptr()) for t in self.hosts + self.slots + [self.other]}
        self.assertFalse([p for p in pointers if p in lines[0]])
        self.assertLess(max(int(v) for v in re.findall(r'=(\d+)', lines[0])), 2 ** 32)

    def test_the_read_follows_the_filter(self):
        # A skipped pair is neither classified on the host pass nor byte-checked: its target's page is nobody's.
        for host in self.hosts:
            host[2] = host[1] + 1
        row = lc.cow_row_pages(dict(pages=dict(runner=self.runner, views=self.views, pairs=[(1, 2)], host_pairs=[])), PAGE_TOKENS + 5)
        self.assertEqual((row['verify_pairs'], row['verify_unequal']), (0, 0))
        self.assertFalse([k for k in row if k.startswith('h_')])
        self.assertTrue([k for k in row if k.startswith('k_')])

    def test_a_failed_read_is_one_count_and_the_row_stays(self):
        row = lc.cow_row_pages(dict(pairs=1, pages=dict(runner=object(), views=None, pairs=[(1, 2)])), 5)
        self.assertEqual(row, dict(pairs=1, pages_error=1))

    def test_a_row_with_no_pairs_reads_nothing(self):
        self.assertEqual(lc.cow_row_pages(dict(zero_ids=1, pages=dict(runner=object(), views=None, pairs=[])), 5), dict(zero_ids=1))


class Receipt(unittest.TestCase):
    def bridge(self):
        return types.SimpleNamespace(rank=2)

    def tx(self):
        return types.SimpleNamespace(txn_seq=66, chunk_start=65296, chunk_end=65444)

    def test_the_line_names_the_rank_chunk_and_times(self):
        lc._COW_ROW[0] = dict(marks=[Mark(at=0.0), Mark(at=1.0), Mark(at=3.0), Mark(at=3.0)], zero_ids=0, pairs=1, fast=1, views=2, bytes=64,
                              host_views=0, zero_host_us=1, copy_host_us=2, host_view_us=0)
        lines = []
        with mock.patch.dict(os.environ, {'FLASH_S5_INWAIT': '1'}), \
                mock.patch('native_measurement.emit_receipt_kv', lambda line, logger: lines.append(line)):
            wb.NativeWorkerBridge._cow_receipt(self.bridge(), self.tx())
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith('flash-s5 cow rank=2 txn_seq=66 chunk=65296-65444 '), lines[0])
        self.assertIn('copy_dev_us=2000', lines[0])
        self.assertIn('fast=1', lines[0])
        self.assertNotIn('marks', lines[0])

    def test_no_row_no_line_and_a_failure_is_counted(self):
        lc.take_cow_row()
        lines = []
        with mock.patch.dict(os.environ, {'FLASH_S5_INWAIT': '1'}), \
                mock.patch('native_measurement.emit_receipt_kv', lambda line, logger: lines.append(line)):
            wb.NativeWorkerBridge._cow_receipt(self.bridge(), self.tx())
            self.assertEqual(lines, [])
            before = lc.COW_FAST['row_errors']
            lc._COW_ROW[0] = dict(marks=[object(), object(), object(), object()])
            wb.NativeWorkerBridge._cow_receipt(self.bridge(), self.tx())
        self.assertEqual(lc.COW_FAST['row_errors'], before + 1)
        self.assertEqual(lines, [])

    def test_flag_off_touches_nothing(self):
        lc._COW_ROW[0] = dict(marks=[])
        self.addCleanup(lc.take_cow_row)
        with mock.patch.dict(os.environ, {'FLASH_S5_INWAIT': '0'}):
            wb.NativeWorkerBridge._cow_receipt(self.bridge(), self.tx())
        self.assertIsNotNone(lc._COW_ROW[0])


MIB = 1024 * 1024


class Cap(unittest.TestCase):
    def setUp(self):
        saved = dict(lc.RESERVED_CAP)
        self.addCleanup(lambda: lc.RESERVED_CAP.update(saved))
        lc.RESERVED_CAP.update(selected=False, cap_bytes=0, fraction=0.0, flushes=0)

    def test_the_cap_from_this_box_ranks(self):
        # Rank 0 and rank 1 of build-3b's record: total 24124 MiB, outside the allocator 1206.5 and 1318.5.
        total = 25296044032
        for other in (1206.5, 1318.5):
            reserved = 21856 * MIB
            free = total - reserved - int(other * MIB)
            cap = lc.reserved_cap_bytes(total, free, reserved)
            self.assertEqual(total - int(other * MIB) - cap, 512 * MIB)

    def test_out_of_range_is_refused(self):
        with self.assertRaises(lc.LaneChunkError):
            lc.reserved_cap_bytes(100, 50, 60)
        with self.assertRaises(lc.LaneChunkError):
            lc.reserved_cap_bytes(100 * MIB, 90 * MIB, 5 * MIB)

    def fake_cuda(self):
        calls = []
        cuda = types.SimpleNamespace(
            current_device=lambda: 3, synchronize=lambda: calls.append('sync'), empty_cache=lambda: calls.append('flush'),
            mem_get_info=lambda device: (2000 * MIB, 24000 * MIB), memory_reserved=lambda device: 21000 * MIB,
            set_per_process_memory_fraction=lambda fraction, device: calls.append((round(fraction, 6), device)))
        return cuda, calls

    def test_on_flushes_then_caps_and_off_lifts_it(self):
        cuda, calls = self.fake_cuda()
        with mock.patch.object(torch, 'cuda', cuda):
            state = lc.select_reserved_cap(True)
            self.assertEqual(calls[:2], ['sync', 'flush'])
            # 1000 MiB outside the allocator: at most 22488 reserved keeps 512 free.
            self.assertEqual(state['cap_bytes'], 22488 * MIB)
            self.assertEqual(calls[2], (round(22488 / 24000, 6), 3))
            self.assertEqual(state['flushes'], 1)
            self.assertEqual(lc.select_reserved_cap(True)['flushes'], 1, 'on to on does nothing')
            self.assertEqual(len(calls), 3)
            state = lc.select_reserved_cap(False)
            self.assertEqual(calls[3], (1.0, 3))
            self.assertEqual((state['selected'], state['cap_bytes']), (False, 0))
            lc.select_reserved_cap(False)
            self.assertEqual(len(calls), 4, 'off to off touches nothing')

    def test_not_bool_is_refused(self):
        for select in (lc.select_reserved_cap, lc.select_cow_fast):
            with self.assertRaises(lc.LaneChunkError):
                select(1)


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    unittest.main()


class ThroughTheWriter(unittest.TestCase):
    """build-5b: build-4c lost every cow line in the receipt writer's allow-list, and the tests above mock the writer away. Here the real writer and the real checker."""

    def setUp(self):
        import native_measurement as nm
        import receipt_lines as rl
        self.nm, self.rl, self.out = nm, rl, []
        self.writer = nm.ReceiptWriter(fd=9, write=lambda fd, data: (self.out.append(data), len(data))[1], getpid=lambda: 77)
        with mock.patch.object(nm, 'receipt_path', lambda fd, limit=None: 'atomic-fifo'):
            self.writer._start()
        patcher = mock.patch.object(nm, '_RECEIPT_WRITER', self.writer)
        patcher.start()
        self.addCleanup(patcher.stop)
        lc.COW_FAST.update(row_errors=0, rows=0)
        self.addCleanup(lc.COW_FAST.update, row_errors=0, rows=0)
        self.addCleanup(lc._COW_ROW.__setitem__, 0, None)

    def receipt(self):
        lc._COW_ROW[0] = dict(marks=[Mark(at=0.0), Mark(at=1.0), Mark(at=3.0), Mark(at=7.0)], zero_ids=0, pairs=37, fast=37, views=6, bytes=64,
                              host_views=6, host_fast=222, host_bytes=128, zero_host_us=1, copy_host_us=2, host_view_us=3)
        with mock.patch.dict(os.environ, {'FLASH_S5_INWAIT': '1'}):
            wb.NativeWorkerBridge._cow_receipt(types.SimpleNamespace(rank=2), types.SimpleNamespace(txn_seq=66, chunk_start=65296, chunk_end=65444))

    def test_the_line_is_written_and_the_checker_reads_it(self):
        before = self.writer.seq
        self.receipt()
        self.assertEqual(self.writer.faults, 0)
        self.assertEqual(self.writer.seq, before + 1)
        line = self.out[-1].decode()
        self.assertTrue(line.startswith('flash-s5 cow rank=2 txn_seq=66 chunk=65296-65444 '), line)
        self.assertIn('host_view_dev_us=4000', line)
        record = self.rl.parse_line(line)
        self.assertEqual(record['kind'], 'cow')
        self.assertEqual(record['rank'], 2)
        self.assertEqual(record['sha'], self.rl.sha16([record['text']]))
        self.assertEqual((record['pid'], record['seq']), (77, before))
        self.assertEqual(lc.COW_FAST['row_errors'], 0)

    def test_a_refused_line_counts_a_row_error(self):
        with mock.patch.object(self.nm, 'RECEIPT_KV_PREFIXES', ('flash-s5 memory ',)):
            self.receipt()
        self.assertEqual(self.writer.faults, 1)
        self.assertEqual(lc.COW_FAST['row_errors'], 1)
