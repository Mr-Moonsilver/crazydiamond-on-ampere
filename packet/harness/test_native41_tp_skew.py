"""Native41 held-e: auto-arm-only per-chunk allocator receipts on each worker.

The TP completion-skew tests that used to live here moved, unchanged in
composition, to the trunk-named ``test_e3_tp_finalize_skew``. They now ship
with the engine fix as one standalone commit that native43 and the trunk can
cherry-pick. This file keeps the native41-only receipts. They run over the
same connected composition, ``SkewNetwork``, imported from that file.

Declared CPU leaf (``native41_fixture_leaves.json``, memory-receipt-cuda):
``FakeCuda`` stands in for torch.cuda's host allocator and driver queries
(``is_initialized``, ``memory_stats``, ``mem_get_info``) on one rank. Those
calls are read-only host queries on a GPU, and the fake returns per-rank
values in the same shape. No GPU, box or performance claim.
"""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'harness'), str(ROOT / 'patches/s5_engine')]
import test_e3_native_schedule_profile as sp  # noqa: E402
import test_e3_native_submission_credit as credit  # noqa: E402
from test_e3_tp_finalize_skew import SkewNetwork  # noqa: E402

class FakeCuda:
    """Leaf for torch.cuda's host allocator/driver queries on one rank."""

    def __init__(self, rank, initialized=True, fail=False):
        self.rank = rank
        self.initialized = initialized
        self.fail = fail
        self.calls = 0

    def is_initialized(self):
        return self.initialized

    def memory_stats(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError('allocator query failed')
        base = 1000 * (self.rank + 1)
        return {'allocated_bytes.all.current': base + self.calls,
                'reserved_bytes.all.current': base + 500,
                'inactive_split_bytes.all.current': 7 + self.rank,
                'num_alloc_retries': self.calls,
                'num_ooms': 0}

    def mem_get_info(self):
        return (300 + self.rank, 24 * 1024)


class AutoArmMemoryReceipts(credit.CreditHarness, unittest.TestCase):
    """Per-rank allocator receipts at each chunk open and after its drained
    release, only with FLASH_S5_AUTO_ARM=1, never a control input."""
    legacy_schedule_source = False
    query_rows = 64
    stack_type = sp.HistoryStack
    selection_counts = (4097, 4097)

    @classmethod
    def setUpClass(cls):
        if not credit.REFERENCE.is_dir():
            raise RuntimeError('required reference absent; connected proof NOT_RUN')
        super().setUpClass()

    def network(self, switch, **cuda):
        with mock.patch.dict(os.environ, {'FLASH_S5_AUTO_ARM': switch}):
            network = SkewNetwork(self)
        bridges = {rank: (network.qsa[rank].bridge if rank in network.qsa
                          else network.workers[rank]._flash_s5_bridge) for rank in range(4)}
        fakes = {}
        for rank, bridge in bridges.items():
            fakes[rank] = bridge._memory_cuda = FakeCuda(rank, **cuda)
        return network, bridges, fakes

    def finish(self, network):
        for _ in range(600):
            network.complete()
            network.driver.advance(deliver_outputs=False)
            if not network.driver.active:
                break
        self.assertIsNone(network.driver.error)
        self.assertFalse(network.driver.active)
        self.assertTrue(network.all_dispatchers_reset())

    def test_switch_on_records_open_and_released_per_rank(self):
        network, bridges, fakes = self.network('1')
        tx = network.record.transaction
        with self.assertLogs('worker_bridge', level='INFO') as logs:
            self.finish(network)
        for rank, bridge in bridges.items():
            rows = bridge.memory_rows
            self.assertEqual([row['point'] for row in rows], ['open', 'released'])
            self.assertEqual(bridge.memory_receipt_errors, 0)
            for row in rows:
                self.assertEqual((row['rank'], row['txn_seq'], row['chunk_start'], row['chunk_end']),
                                 (rank, tx.txn_seq, tx.chunk_start, tx.chunk_end))
                self.assertEqual(set(row), {'point', 'rank', 'txn_seq', 'chunk_start', 'chunk_end',
                                            'allocated', 'reserved', 'inactive_split',
                                            'alloc_retries', 'ooms', 'free', 'total',
                                            'sync_all_streams', 'device_allocs', 'device_frees'})
                self.assertEqual((row['free'], row['total']), (300 + rank, 24 * 1024))
                self.assertEqual(row['inactive_split'], 7 + rank)
            self.assertEqual(fakes[rank].calls, 2)
        lines = [line for line in logs.output if 'flash-s5 memory ' in line]
        self.assertEqual(len(lines), 8)
        self.assertTrue(all(len(line) < 400 for line in lines))

    def test_switch_off_never_queries_and_keeps_the_same_commands(self):
        on, _, _ = self.network('1')
        self.finish(on)
        off, bridges, fakes = self.network('0')
        self.finish(off)
        for rank, bridge in bridges.items():
            self.assertEqual(bridge.memory_rows, [])
            self.assertEqual(fakes[rank].calls, 0)
        # The receipt changes no RPC: both runs send the same method sequence.
        self.assertEqual(on.methods(), off.methods())

    def test_uninitialized_or_failing_cuda_is_counted_never_raised(self):
        network, bridges, fakes = self.network('1', initialized=False)
        self.finish(network)
        for rank, bridge in bridges.items():
            self.assertEqual((bridge.memory_rows, bridge.memory_receipt_errors, fakes[rank].calls),
                             ([], 0, 0))
        network, bridges, fakes = self.network('1', fail=True)
        self.finish(network)
        for bridge in bridges.values():
            self.assertEqual(bridge.memory_rows, [])
            self.assertEqual(bridge.memory_receipt_errors, 2)


if __name__ == '__main__':
    unittest.main()
