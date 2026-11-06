#!/usr/bin/env python3
"""coverage-soak-fix build-2 (lead, 6 Nov 08:50): the Worker_PP RSS owner, named and turned off. Hub, no torch, no GPU.

The soak's worker RSS grew only on nr, about 1.5-2.0 GiB per worker per arm, and stayed. The owner is the CoW byte check at
release (``lane_chunk.cow_host_verify``, from ``worker_bridge._cow_receipt``). For every host CoW pair it reads the src and
dst page prefix of every local QSA layer through ``_qsa_host_kv``, the registered host pool, on the CPU. The pool is mmap'd
MAP_SHARED|MAP_ANONYMOUS (``host_memory.allocate_registered_host_tensor``) and is not in a worker's RSS until the CPU touches it:
6 x 5.173 GiB per worker against worker RSS of 2.8-25.7 GB. A CPU read maps those pages into the worker for good. Only a native
append's first chunk has host CoW pairs: no fresh turn, and nothing on ny.

This file proves what the hub can:
- the compose-5 table selects the check off on every arm and the pair stays matched;
- with the check off, the release path reads no host-pool page; on, it reads both prefixes on every layer (the control);
- the mechanism itself: a CPU read of an untouched shared anonymous mapping raises the process's RssShmem by the bytes read, and
  a re-read adds nothing.
The box still has to show worker RSS flat on nr; the grade's worker RSS FAIL is kept for that.
"""
import mmap
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'patches/s5_engine')]

import compose5_arms as T  # noqa: E402
import lane_chunk as lc  # noqa: E402

PAGE_TOKENS = 4048
LAYERS = 6


class Arms(unittest.TestCase):
    def test_every_arm_selects_the_check_off(self):
        self.assertIs(T.COW_VERIFY, False)
        rows = T.arms()
        self.assertTrue(rows)
        self.assertEqual({r['label']: r['cow_verify'] for r in rows}, {r['label']: False for r in rows})

    def test_the_pairs_stay_matched(self):
        T.check()
        rows = T.by_label()
        for index in T.REPLAY_SEEDS:
            self.assertIs(rows['nt.g2-live-%d' % index]['cow_verify'], rows['ny.g2-live-%d' % index]['cow_verify'])

    def test_the_settings_leave_the_runner_check_off(self):
        settings = (HERE / 'runs/q18a_compose_throughput_settings.sh').read_text()
        self.assertIn('QSA_COW_VERIFY=-', settings)


class _Page:
    """One QSA host pool's ``host[block, :, :prefix]`` slice: records the read."""

    def __init__(self, pool, key):
        self.pool, self.key = pool, key

    def view(self, _dtype):
        self.pool.reads.append(self.key)
        return self


class _Pool:
    """A layer's ``_qsa_host_kv`` [blocks, 1, 4048, 512]: indexing records which block and how many rows the CPU read."""

    shape = (16, 1, PAGE_TOKENS, 512)

    def __init__(self):
        self.reads = []

    def __getitem__(self, key):
        block, _heads, rows = key
        return _Page(self, (block, rows.stop))


def _runner():
    pools = [_Pool() for _ in range(LAYERS)]
    layers = {'layer.%d' % i: NS(_qsa_host_kv=pool) for i, pool in enumerate(pools)}
    layers['mamba'] = NS()
    return NS(compilation_config=NS(static_forward_context=layers), kv_caches=[], kv_cache_config=NS(num_blocks=16)), pools


class ReleasePath(unittest.TestCase):
    """``cow_row_pages`` as ``worker_bridge._cow_receipt`` calls it at release, with the lane drained."""

    def setUp(self):
        self._saved = dict(lc.COW_VERIFY)
        self.addCleanup(lc.COW_VERIFY.update, self._saved)
        self.runner, self.pools = _runner()
        self.torch = types.ModuleType('torch')
        self.torch.uint8 = 'uint8'
        self.torch.equal = lambda a, b: True

    def row(self):
        return dict(zero_ids=1, pages=dict(runner=self.runner, views=None, pairs=[(3, 7)], host_pairs=[(3, 7)]))

    def test_off_reads_no_host_page(self):
        lc.select_cow_verify(False)
        lc.COW_VERIFY['skipped'] = 0
        with mock.patch.dict(sys.modules, {'torch': self.torch}), \
                mock.patch.object(lc, 'cow_residency', side_effect=AssertionError('residency ran')):
            row = lc.cow_row_pages(self.row(), 5 * PAGE_TOKENS + 1584)
        self.assertEqual([pool.reads for pool in self.pools], [[]] * LAYERS)
        self.assertEqual((lc.COW_VERIFY['skipped'], 'verify_pairs' in row, 'pages' in row), (1, False, False))

    def test_on_reads_both_prefixes_on_every_layer(self):
        # the control: the soak's selection; this is the read that mapped the pool pages
        lc.select_cow_verify(True)
        with mock.patch.dict(sys.modules, {'torch': self.torch}), \
                mock.patch.object(lc, 'cow_residency', return_value={}):
            row = lc.cow_row_pages(self.row(), 5 * PAGE_TOKENS + 1584)
        self.assertEqual(row.get('verify_pairs'), LAYERS, row)
        self.assertEqual(row.get('verify_prefix'), 1584)
        self.assertEqual([sorted(pool.reads) for pool in self.pools], [[(3, 1584), (7, 1584)]] * LAYERS)


def _shmem_kib():
    for line in open('/proc/self/status'):
        if line.startswith('RssShmem:'):
            return int(line.split()[1])
    raise unittest.SkipTest('no RssShmem in /proc/self/status')


class Mechanism(unittest.TestCase):
    """What ``allocate_registered_host_tensor`` maps (``mmap.mmap(-1, n)``), without CUDA: reads map pages, and they stay."""

    BLOCK = PAGE_TOKENS * 1024          # one pool block of one layer: 4048 tokens x 1 KiB

    def test_a_cpu_read_maps_what_it_reads_and_it_stays(self):
        blocks = 8
        mapping = mmap.mmap(-1, blocks * self.BLOCK)
        self.addCleanup(mapping.close)
        view = memoryview(mapping)
        self.addCleanup(view.release)
        start = _shmem_kib()
        self.assertLess(start, 1024, 'a fresh mapping starts unmapped')
        prefix = 1584 * 1024
        for block in (1, 5):
            bytes(view[block * self.BLOCK:block * self.BLOCK + prefix])
        grown = _shmem_kib() - start
        self.assertGreaterEqual(grown, 2 * prefix // 1024)
        self.assertLess(grown, 2 * prefix // 1024 + 4096, 'about what was read, not the mapping')
        bytes(view[self.BLOCK:self.BLOCK + prefix])
        self.assertEqual(_shmem_kib() - start, grown, 'a re-read maps nothing new')

    def test_no_read_maps_nothing(self):
        mapping = mmap.mmap(-1, 8 * self.BLOCK)
        self.addCleanup(mapping.close)
        start = _shmem_kib()
        self.assertLess(_shmem_kib() - start, 64)


if __name__ == '__main__':
    unittest.main()
