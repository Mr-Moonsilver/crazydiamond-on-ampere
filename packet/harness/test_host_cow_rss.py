#!/usr/bin/env python3
"""coverage-soak-fix build-3 (lead, 6 Nov 11:45): the QSA host CoW through stock's copy, and the box host's memory beside worker RssShmem.

Build-2 turned the CPU byte check off, and nr's Worker_PP RssShmem still grew, 26-32 MiB per host CoW pair per worker
(READING-soak-fix-2.md). The remaining toucher: ``_apply_allocated_pages`` sent the QSA host views (UVA views of the registered pool)
through ``copy_blocks_fast``, one host-to-host ``copy_`` per block, which ran on the CPU (host_view_us p50 9.6 ms a pair) and mapped
the src and dst pool pages into the worker. Stock's runner copies the same views with its indexed copy on the GPU, and ny stayed flat.
Build-3 always sends the host views through ``ports.copy_blocks_inplace``; the device copy keeps its fast path.

The record could not tell new host memory from pages already charged and newly mapped: one memory.current/MemAvailable point, no smaps.
Build-3 reads the box host's meminfo and cgroup at each replay arm's start and end and puts the growth on the RSS line.

Hub, no torch, no GPU. The container's test_inwait_cow covers the same path on real torch tensors with stock's real copy.
"""
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'patches/s5_engine')]

import lane_chunk as lc  # noqa: E402
import e3_compose5_box as box  # noqa: E402
import test_compose5_box as tb  # noqa: E402

VIEWS = ['host-view-%d' % i for i in range(6)]


class HostCopy(unittest.TestCase):
    """``LaneRequest._apply_allocated_pages`` with fakes: which copy gets which caches."""

    def setUp(self):
        saved = {name: dict(getattr(lc, name)) for name in ('COW_FAST', 'COW_HOST_FILTER')}
        for name, value in saved.items():
            self.addCleanup(getattr(lc, name).update, value)
        self.addCleanup(lc.take_cow_row)
        self.log = []
        self.runner = NS(kv_caches='device-caches', kv_cache_config=NS(num_blocks=16), _qsa_host_views=list(VIEWS),
                         kv_block_zeroer=NS(zero_block_ids=lambda ids: self.log.append(('zero', tuple(ids)))))
        self.ports = NS(block_copy=lambda src, dst: (src, dst), cow_mark=None,
                        copy_blocks_inplace=lambda caches, n, copies: self.log.append(('stock', caches, tuple(copies))))
        self.request = lc.LaneRequest.__new__(lc.LaneRequest)
        self.request.runner, self.request.ports = self.runner, self.ports

    def fast(self):
        def copy(caches, num_blocks, pairs):
            self.log.append(('fast', caches, tuple(pairs)))
            return dict(views=1, bytes=8 * len(pairs))
        return mock.patch.object(lc, 'copy_blocks_fast', side_effect=copy)

    def apply(self, pairs=((3, 7),), zero=()):
        self.request._apply_allocated_pages(NS(new_block_ids_to_zero=zero, cow_pairs=pairs, block_tables=((7,),)))
        return lc.take_cow_row()

    def test_fast_on_copies_the_device_fast_and_the_host_views_with_stock(self):
        lc.select_cow_fast(True)
        with self.fast():
            row = self.apply()
        self.assertEqual(self.log, [('fast', 'device-caches', ((3, 7),)), ('stock', VIEWS, ((3, 7),))])
        self.assertEqual((row['fast'], row['host_fast'], row['host_bytes'], row['host_pairs'], row['host_views']), (1, 0, None, 1, 6))

    def test_no_host_view_ever_reaches_the_fast_copy(self):
        lc.select_cow_fast(True)
        for pairs in (((3, 7),), ((1, 2), (4, 5), (9, 11)), ((1, 2), (2, 3))):
            with self.subTest(pairs=pairs):
                self.log.clear()
                with self.fast():
                    self.apply(pairs=pairs)
                self.assertEqual([entry[1] for entry in self.log if entry[0] == 'fast'], ['device-caches'])
                self.assertEqual([entry[2] for entry in self.log if entry[0] == 'stock' and entry[1] == VIEWS], [pairs])

    def test_a_device_fallback_still_copies_the_host_views_once_with_stock(self):
        lc.select_cow_fast(True)
        # the fast copy refuses a chain (returns None): the device goes to stock, the host views to stock, once each
        with mock.patch.object(lc, 'copy_blocks_fast', side_effect=lambda c, n, p: self.log.append(('fast', c, tuple(p)))):
            row = self.apply(pairs=((1, 2), (2, 3)))
        self.assertEqual(self.log, [('fast', 'device-caches', ((1, 2), (2, 3))),
                                    ('stock', 'device-caches', ((1, 2), (2, 3))), ('stock', VIEWS, ((1, 2), (2, 3)))])
        self.assertEqual((row['fast'], row['host_fast']), (0, 0))

    def test_fast_off_copies_both_with_stock(self):
        lc.select_cow_fast(False)
        with self.fast():
            row = self.apply()
        self.assertEqual(self.log, [('stock', 'device-caches', ((3, 7),)), ('stock', VIEWS, ((3, 7),))])
        self.assertEqual((row['fast'], row['host_fast']), (0, 0))

    def test_a_filtered_out_host_pair_copies_no_host_view(self):
        lc.select_cow_fast(True)
        lc.select_cow_host_filter(True)
        self.runner._qsa_host_groups = [0]
        with self.fast():
            self.request._apply_allocated_pages(NS(new_block_ids_to_zero=(), cow_pairs=((3, 7),), block_tables=((9,), (7,))))
        self.assertEqual(self.log, [('fast', 'device-caches', ((3, 7),))])

    def test_no_pages_no_copy(self):
        lc.select_cow_fast(True)
        with self.fast():
            self.request._apply_allocated_pages(NS(new_block_ids_to_zero=(), cow_pairs=(), block_tables=()))
        self.assertEqual(self.log, [])


MEMINFO = 'MemTotal:       263842036 kB\nMemFree:          1234567 kB\nMemAvailable:   14524789 kB\nShmem:          201234567 kB\n'
STAT = 'anon 41234567890\nfile 9876543210\nkernel 123\nshmem 206064177152\nfile_mapped 1\n'


def opener(files):
    def open_(path, *_a, **_k):
        if path not in files:
            raise FileNotFoundError(path)
        return io.StringIO(files[path])
    return open_


def box_files(current=257451376640, available_kib=14524789, shmem_kib=201234567, cgroup_shmem=206064177152, limit='max'):
    return {'/proc/meminfo': MEMINFO.replace('14524789', str(available_kib)).replace('201234567', str(shmem_kib)),
            '/sys/fs/cgroup/memory.current': '%d\n' % current, '/sys/fs/cgroup/memory.max': '%s\n' % limit,
            '/sys/fs/cgroup/memory.stat': STAT.replace('206064177152', str(cgroup_shmem))}


class HostMemory(unittest.TestCase):
    def test_reads_meminfo_and_the_cgroup(self):
        out = box.host_memory(opener=opener(box_files()))
        self.assertEqual(out, dict(errors=[], memtotal_kib=263842036, memavailable_kib=14524789, shmem_kib=201234567,
                                   memory_current=257451376640, memory_max=None, cgroup_anon=41234567890, cgroup_file=9876543210,
                                   cgroup_shmem=206064177152))

    def test_a_limit_is_read_as_bytes(self):
        self.assertEqual(box.host_memory(opener=opener(box_files(limit='274877906944')))['memory_max'], 274877906944)

    def test_a_missing_file_is_an_error_never_a_raise(self):
        out = box.host_memory(opener=opener({'/proc/meminfo': MEMINFO}))
        self.assertEqual(out['errors'], ['memory.current:FileNotFoundError', 'memory.max:FileNotFoundError', 'memory.stat:FileNotFoundError'])
        self.assertEqual((out['memory_current'], out['memavailable_kib']), (None, 14524789))

    def test_engine_memory_carries_the_host_and_survives_its_failure(self):
        def run(cmd, **kw):
            return mock.Mock(returncode=0, stdout='0, 23112, 24576\n' if cmd[0] == 'nvidia-smi' else 'PID RSS COMMAND\n4101 812345 python3\n')
        out = box.engine_memory(run=run, host=lambda: dict(errors=[], memory_current=5))
        self.assertEqual((out['schema'], out['host'], out['errors']), ('soak-memory/2', dict(errors=[], memory_current=5), []))

        def broken():
            raise OSError('no proc')
        out = box.engine_memory(run=run, host=broken)
        self.assertEqual((out['host'], out['errors'], out['rss_kib']), (None, ['host:OSError'], 812345))

    def test_the_default_reads_this_hosts_files(self):
        out = box.host_memory()       # the hub's own: read only
        self.assertIsInstance(out.get('memavailable_kib'), int, out)


def with_host(mem, **files):
    return dict(mem, host=box.host_memory(opener=opener(box_files(**files))))


class RssLine(unittest.TestCase):
    """The RSS line: worker RssShmem growth beside the host's memory.current, MemAvailable, Shmem and the cgroup's shmem."""

    def grade(self, mem):
        return tb.Grade().grade_of(tb.Soak().arms(mem=mem))

    def report(self, result):
        with box._bound():
            return box.report(result)

    def test_mapping_only_reads_flat_host_beside_grown_workers(self):
        mem = {tb.R1: dict(memory=dict(start=with_host(tb.memory()), end=with_host(tb.memory(grow=(1320, 1270, 1208, 1151)))))}
        result = self.grade(mem)
        host = result['memory'][tb.R1]['host']
        self.assertEqual((host['memory_current']['growth'], host['memavailable_kib']['growth'], host['cgroup_shmem']['growth']), (0, 0, 0))
        line = [l for l in self.report(result).splitlines() if l.startswith('- %s worker RSS:' % tb.R1)][0]
        self.assertIn('worker-rss-up-pid4210-1320MiB', line, 'the FAIL is kept')
        self.assertIn('worker RssShmem MiB pid4210 +660, pid4211 +635, pid4212 +604, pid4213 +575', line)
        self.assertIn('host MiB: cgroup memory.current +0, MemAvailable +0, meminfo Shmem +0, cgroup shmem +0', line)

    def test_new_memory_reads_on_the_host_side(self):
        start = with_host(tb.memory())
        end = with_host(tb.memory(grow=(1024, 0, 0, 0)), current=257451376640 + (1024 << 20), available_kib=14524789 - (1024 << 10),
                        shmem_kib=201234567 + (1024 << 10), cgroup_shmem=206064177152 + (1024 << 20))
        result = self.grade({tb.R1: dict(memory=dict(start=start, end=end))})
        line = [l for l in self.report(result).splitlines() if l.startswith('- %s worker RSS:' % tb.R1)][0]
        self.assertIn('host MiB: cgroup memory.current +1024, MemAvailable -1024, meminfo Shmem +1024, cgroup shmem +1024', line)

    def test_an_unread_host_side_says_so(self):
        mem = {tb.R1: dict(memory=dict(start=tb.memory(), end=with_host(tb.memory())))}
        line = [l for l in self.report(self.grade(mem)).splitlines() if l.startswith('- %s worker RSS:' % tb.R1)][0]
        self.assertIn('host not read', line)
        self.assertIsNone(self.grade(mem)['memory'][tb.R1]['host'])

    def test_the_ny_control_gets_the_same_read_ungraded(self):
        mem = {tb.W1: dict(memory=dict(start=with_host(tb.memory()), end=with_host(tb.memory(grow=(64, 0, 0, 0)))))}
        result = self.grade(mem)
        self.assertNotIn(tb.W1, result['worker_rss'])
        text = self.report(result)
        self.assertIn('- %s worker RSS (read, not graded): worker RssShmem MiB pid4210 +32' % tb.W1, text)
        self.assertNotIn('nr-first-', ''.join(l for l in text.splitlines() if 'worker RSS (read' in l))


if __name__ == '__main__':
    unittest.main()
