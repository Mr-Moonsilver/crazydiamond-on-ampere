"""Exhibit the TP tail asymmetry left open by a pending final sparse event.

The model and communicator are explicit CPU leaves. This does not execute an
NCCL collective or prove its timeout/abort behavior on multiple GPUs.
"""
from collections import Counter
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'harness'), str(ROOT / 'patches/s5_engine')]
import test_e3_native_schedule_profile as profile
import qsa_driver


class CountingCollective:
    """Nonblocking model leaf recording entry to the production o_proj seam."""
    def __init__(self):
        self.calls = Counter()

    def bind(self, attention, rank):
        original = attention._flash_e3_attention_post

        def recorded(state):
            self.calls[rank] += 1
            return original(state)

        # Only a synthetic fixture instance is wrapped. Runtime modules and
        # the actual QsaLayerDriver method are unchanged.
        attention._flash_e3_attention_post = recorded


class TailSkew(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        profile.ScheduleProfile.setUpClass()

    def test_one_peer_enters_tp_tail_while_other_final_event_is_pending(self):
        case = profile.ScheduleProfile('runTest')
        case.setUp()
        self.addCleanup(case.doCleanups)
        network = profile.ProfileNetwork(case)
        collective = CountingCollective()
        for rank, stack in network.qsa.items():
            collective.bind(stack.attention, rank)
        for tick in range(28):
            network.complete_everything()
            network.driver.advance(deliver_outputs=False)
        self.assertEqual([network.qsa[r].operation.stats['copy_events'] for r in (0, 1)], [17, 17])
        self.assertEqual(dict(collective.calls), {})
        network.complete_everything()
        # Hold only the last consumer event of TP peer 1. Peer 0 has identical
        # queried schedule profile but a completed consumer event.
        network.qsa[1].device.events[-1].ready = False
        network.driver.advance(deliver_outputs=False)
        self.assertEqual([collective.calls[r] for r in (0, 1)], [1, 0])
        self.assertTrue(network.driver._inflight)
        self.assertIsNone(network.driver.stop_kind)
        # The generated production post path enters RowParallelLinear.o_proj;
        # its default result reduction is a TP collective. CPU fake returns
        # immediately, so liveness of the real collective remains unproved.
        self.assertIn('self.o_proj(flat_output)', qsa_driver._ATTN_METHODS)
        pinned = (ROOT / 'reference/vllm-src/models/qwen4_exp/nvidia/qsa.py').read_text()
        self.assertIn('reduce_results: bool = True', pinned)
        self.assertIn('self.o_proj = RowParallelLinear(', pinned)


if __name__ == '__main__':
    unittest.main()
