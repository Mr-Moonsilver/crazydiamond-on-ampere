"""Decode-phase dp5c: the connected host proof at a native chunk's real rows, 1024 the control.

test_e3_native_qsa_fused's real NativeSegmentDriver, bridge, dispatcher, QsaLayerDriver, QsaPrefillOperation and generated attention, over its declared CPU leaves,
which until now ran at query_rows 64. The box's fused route, stage by stage. Not a hardware certificate: no device order or bytes here.
"""
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'harness'), str(ROOT / 'patches/s5_engine')]
import test_e3_native_pp_stop_order as pp
import test_e3_native_qsa_fused as f


def at(rows):
    made = {}
    for base in (f.OneBoundedCommand, f.PeerSkew, f.Cancel, f.Commitment, f.Refusal, f.Selection, f.StopBound, f.Burst,
                 pp.PPStopOrder):
        name = f'{base.__name__}At{rows}'
        made[name] = type(name, (base,), dict(query_rows=rows, __module__=__name__))
    return made


for _rows in (1024, 2048):
    globals().update(at(_rows))


class Witness(unittest.TestCase):
    """query_rows has to reach the real operation, or every case above proves nothing about the size."""

    def test_the_operation_sees_the_rows(self):
        for rows in (1024, 2048):
            case = globals()[f'OneBoundedCommandAt{rows}']('test_multi_bucket_layer_is_one_command_with_every_device_event_unresolved')
            type(case).setUpClass()
            self.addCleanup(type(case).tearDownClass)
            case.setUp()
            self.addCleanup(case.doCleanups)
            seen = []
            original = f.qo.QsaPrefillOperation.observe_chunk_shape

            def observe(operation, observed, min_rows, route, _original=original):
                seen.append(observed)
                return _original(operation, observed, min_rows, route)
            with mock.patch.object(f.qo.QsaPrefillOperation, 'observe_chunk_shape', observe):
                network = f.FusedNetwork(case)
                for stack in network.qsa.values():
                    self.assertEqual(stack.tokens, rows)
                    self.assertEqual(stack.main._qsa_host_meta[0], (0, rows))
                    self.assertEqual(stack.attention.topk_indices_buffer.shape[0], rows)
                network.run_until(lambda: network.layer_accepted(0) and network.layer_accepted(1), ready=())
            self.assertEqual(set(seen), {rows})


if __name__ == '__main__':
    unittest.main()
