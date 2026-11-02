#!/usr/bin/env python3
"""alpha-next per-chunk peak-allocated receipt (REPORT.md option 2; lead 2 Nov 13:30): recorded, never gating (CPU).

The bridge resets the allocator peak at a chunk's open and writes one `flash-s5 memory-peak ` line at its release, both
only where the `flash-s5 memory` receipts already run (FLASH_S5_AUTO_ARM). A failure is counted, never raised; no CUDA
means no line; the line is invisible to every inherited memory grader (they match `flash-s5 memory point=`).
"""
import ast
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
sys.path.insert(0, str(ROOT / 'harness'))
import worker_bridge as wb  # noqa: E402

LINE = re.compile(r'flash-s5 memory-peak rank=(\d+) txn_seq=(\d+) chunk=(\d+)-(\d+) peak_allocated=(-?\d+) '
                  r'peak_reserved=(-?\d+) allocated_at_open=(-?\d+) stage_timer=([01])$')


class PeakCuda:
    """Allocator counters with a peak that only a reset lowers."""

    def __init__(self, initialized=True):
        self.initialized = initialized
        self.allocated = self.reserved = 0
        self.peak_allocated = self.peak_reserved = 0
        self.resets = 0

    def is_initialized(self):
        return self.initialized

    def set(self, allocated, reserved):
        self.allocated, self.reserved = allocated, reserved
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)

    def reset_peak_memory_stats(self):
        self.resets += 1
        self.peak_allocated, self.peak_reserved = self.allocated, self.reserved

    def memory_stats(self):
        return {'allocated_bytes.all.current': self.allocated, 'reserved_bytes.all.current': self.reserved,
                'allocated_bytes.all.peak': self.peak_allocated, 'reserved_bytes.all.peak': self.peak_reserved}


def bridge(cuda, rank=2):
    return SimpleNamespace(rank=rank, _memory_cuda=cuda, memory_receipt_errors=0)


TX = SimpleNamespace(txn_seq=7, chunk_start=44528, chunk_end=45552)


class PeakReceipt(unittest.TestCase):
    def chunk(self, cuda, b, during):
        wb.NativeWorkerBridge._peak_open(b)
        for allocated, reserved in during:
            cuda.set(allocated, reserved)
        with self.assertLogs('worker_bridge', level='INFO') as logs:
            wb.NativeWorkerBridge._peak_receipt(b, TX)
        self.assertEqual(len(logs.records), 1)
        return LINE.search(logs.records[0].getMessage())

    def test_peak_is_per_chunk(self):
        cuda = PeakCuda()
        b = bridge(cuda)
        cuda.set(900, 1000)
        cuda.set(100, 1000)            # an earlier, higher peak that must not leak into the next chunk
        m = self.chunk(cuda, b, [(300, 1000), (250, 1200), (120, 1200)])
        self.assertIsNotNone(m)
        self.assertEqual(tuple(int(x) for x in m.groups()), (2, 7, 44528, 45552, 300, 1200, 100, 0))
        self.assertEqual((cuda.resets, b.memory_receipt_errors), (1, 0))

    def test_stage_timer_marks_a_lower_bound(self):
        cuda = PeakCuda()
        b = bridge(cuda)
        with mock.patch.dict(os.environ, {'QSA_STAGE_TIMING': '1'}):
            m = self.chunk(cuda, b, [(10, 20)])
        self.assertEqual(m.group(8), '1')

    def test_no_cuda_means_no_line_and_no_reset(self):
        cuda = PeakCuda(initialized=False)
        b = bridge(cuda)
        wb.NativeWorkerBridge._peak_open(b)
        with self.assertNoLogs('worker_bridge', level='INFO'):
            wb.NativeWorkerBridge._peak_receipt(b, TX)
        self.assertEqual((cuda.resets, b.memory_receipt_errors), (0, 0))

    def test_failure_is_counted_never_raised(self):
        class Broken(PeakCuda):
            def reset_peak_memory_stats(self):
                raise RuntimeError('x')

            def memory_stats(self):
                raise RuntimeError('y')
        b = bridge(Broken())
        wb.NativeWorkerBridge._peak_open(b)
        wb.NativeWorkerBridge._peak_receipt(b, TX)
        self.assertEqual(b.memory_receipt_errors, 2)
        self.assertIsNone(b._peak_at_open)

    def test_missing_open_reads_minus_one(self):
        cuda = PeakCuda()
        b = bridge(cuda)
        cuda.set(5, 6)
        with self.assertLogs('worker_bridge', level='INFO') as logs:
            wb.NativeWorkerBridge._peak_receipt(b, TX)
        self.assertEqual(LINE.search(logs.records[0].getMessage()).group(7), '-1')


class Placement(unittest.TestCase):
    """Both calls sit inside the bridge's existing `if self._memory_receipts:` blocks, right after the memory receipt."""

    def test_calls_only_under_the_memory_receipt_flag(self):
        tree = ast.parse((ROOT / 'patches/s5_engine/worker_bridge.py').read_text())
        found = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and ast.unparse(node.test) == 'self._memory_receipts':
                calls = [ast.unparse(s) for s in node.body]
                for name in ('self._peak_open()', 'self._peak_receipt(transaction)'):
                    if name in calls:
                        found[name] = calls
        self.assertEqual(found['self._peak_open()'], ["self._memory_receipt('open', record.transaction)",
                                                      'self._peak_open()'])
        self.assertEqual(found['self._peak_receipt(transaction)'], ["self._memory_receipt('released', transaction)",
                                                                    'self._peak_receipt(transaction)'])
        text = (ROOT / 'patches/s5_engine/worker_bridge.py').read_text()
        self.assertEqual(text.count('self._peak_open()'), 1)
        self.assertEqual(text.count('self._peak_receipt(transaction)'), 1)


class InvisibleToInheritedGraders(unittest.TestCase):
    def test_memory_graders_do_not_match(self):
        line = ('flash-s5 memory-peak rank=0 txn_seq=1 chunk=0-1024 peak_allocated=5 peak_reserved=6 '
                'allocated_at_open=1 stage_timer=0')
        import e3_alpha_c0_box as c0
        import e3_alpha_perf_box as ap
        import e3_step_a_box as sa
        for module in (c0, ap, sa):
            self.assertIsNone(module.MEMORY_LINE.search(line), module.__name__)
        self.assertFalse(line.startswith(c0.MEMORY_MARK))
        self.assertNotIn('flash-s5 memory point=', line)


if __name__ == '__main__':
    unittest.main()
