"""TP tail entry waits for both peers' queried final sparse event.

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
import wire_v1 as w


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

    def scenario(self):
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
        network.driver.advance(deliver_outputs=False)
        self.assertEqual([collective.calls[r] for r in (0, 1)], [0, 0])
        self.assertTrue(network.driver._inflight)
        self.assertIsNone(network.driver.stop_kind)
        return network, collective

    def test_abort_before_peer_final_event_keeps_both_out_of_collective(self):
        network, collective = self.scenario()
        network.driver.stop('abort')
        for tick in range(150):
            network.complete_everything()
            network.driver.pump_stop(1)
            if not network.driver.active and not network.driver._inflight:
                break
        self.assertLess(tick, 149)
        self.assertEqual([collective.calls[r] for r in (0, 1)], [0, 0])
        self.assertTrue(network.driver._last_native_closure['complete'])
        self.assertIsNone(network.driver.error)

    def test_peer_final_event_allows_both_tail_entries_before_stop(self):
        network, collective = self.scenario()
        network.complete_everything()
        for tick in range(6):
            network.driver.advance(deliver_outputs=False)
            if all(collective.calls[r] for r in (0, 1)):
                break
        self.assertEqual([collective.calls[r] for r in (0, 1)], [1, 1])
        self.assertEqual(sum(row['kind'] == 'qsa_tail_boundary'
                             for row in network.driver._journal), 1)
        network.driver.stop('abort')
        for tick in range(150):
            network.complete_everything()
            network.driver.pump_stop(1)
            if not network.driver.active and not network.driver._inflight:
                break
        self.assertLess(tick, 149)
        self.assertTrue(network.driver._last_native_closure['complete'])
        self.assertIsNone(network.driver.error)
        # The generated production post path enters RowParallelLinear.o_proj;
        # its default result reduction is a TP collective. CPU fake returns
        # immediately, so liveness of the real collective remains unproved.
        self.assertIn('self.o_proj(flat_output)', qsa_driver._ATTN_METHODS)
        pinned = (ROOT / 'reference/vllm-src/models/qwen4_exp/nvidia/qsa.py').read_text()
        self.assertIn('reduce_results: bool = True', pinned)
        self.assertIn('self.o_proj = RowParallelLinear(', pinned)

    def test_partial_tail_reply_replays_exact_command_before_cancel(self):
        class PendingPeer:
            """Typed endpoint pending leaf; the real peer bridge runs on retry."""
            def __init__(self, worker, driver):
                self.worker = worker
                self.driver = driver
                self.rank = worker.rank
                self.defer = True

            def __getattr__(self, name):
                return getattr(self.worker, name)

            def flash_s5_submit(self, command):
                if (self.defer and self.driver._tail_boundary is not None
                        and command.kind is w.SegmentKind.LAYER
                        and self.worker.rank in command.ranks):
                    return w.SegmentAcceptance(command.transaction, self.worker.rank,
                        command.kind, command.stage, command.op_seq, pending=True)
                return self.worker.flash_s5_submit(command)

        case = profile.ScheduleProfile('runTest')
        case.setUp()
        self.addCleanup(case.doCleanups)
        network = profile.ProfileNetwork(case)
        collective = CountingCollective()
        for rank, stack in network.qsa.items():
            collective.bind(stack.attention, rank)
        endpoint = PendingPeer(network.worker(1), network.driver)
        network.transport.network.workers[1] = endpoint
        for tick in range(100):
            network.complete_everything()
            network.driver.advance(deliver_outputs=False)
            obligation = network.driver._tail_submission
            if obligation is not None and obligation.accepted == {0}:
                break
        else:
            self.fail('partial TP tail submission was not reached')
        self.assertEqual(network.driver.qsa_control.snapshot()['layers'][0]['tail_boundaries'], 1)
        self.assertEqual(obligation.accepted, {0})
        self.assertEqual([collective.calls[r] for r in (0, 1)], [1, 0])
        self.assertIn(obligation.command.op_seq, network.qsa[0].dispatcher.accepted)
        self.assertNotIn(obligation.command.op_seq, network.qsa[1].dispatcher.accepted)
        endpoint.defer = False
        network.driver.stop('abort')
        for tick in range(150):
            network.complete_everything()
            network.driver.pump_stop(1)
            if not network.driver.active and not network.driver._inflight:
                break
        self.assertLess(tick, 149)
        self.assertEqual([collective.calls[r] for r in (0, 1)], [1, 1])
        self.assertEqual(obligation.accepted, {0, 1})
        self.assertEqual(network.driver.qsa_control.snapshot()['layers'][0]['tail_accepted_commands'], 1)
        self.assertTrue(network.driver._last_native_closure['complete'])
        self.assertIsNone(network.driver.error)
        self.assertFalse(network.transport._calls)


if __name__ == '__main__':
    unittest.main()
