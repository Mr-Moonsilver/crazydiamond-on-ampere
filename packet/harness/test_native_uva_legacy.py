"""Legacy nt/na adoption through the real factory and private descriptor owner."""
import types
from unittest import mock
import unittest
import numpy as np
from test_native_uva_retirement import Device, pinned_buffers, nu
import lane_dispatch as ld
import test_e3_lane_chunk as lc
import test_prepare_ahead as pa
import test_parity_ahead as ahead

class LegacyRetirement(unittest.TestCase):
    def setUp(self):
        # parity-n3u: the fake torch has no torch.full, so the case needs the ordinary slot path; SLOTS_FAST is process-global and an
        # earlier suite in the same process can leave it selected (test_inwait_slots, test_parity_ahead): pin it off for this case.
        patcher = mock.patch.dict(lc.lane_chunk.SLOTS_FAST, {'selected': False})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pinned_factory_installs_owner_and_all_three_publications_require_scope(self):
        calls = []
        torch = lc.fake_torch()
        runner = lc.fake_runner(torch, calls)
        d = Device()
        b = pinned_buffers(d, depth=3)
        states, tables = runner.req_states, runner.block_tables
        states.__class__ = type('PinnedRequestState', (type(states),), {'__module__': 'vllm.v1.worker.gpu.states'})
        tables.__class__ = type('PinnedBlockTables', (type(tables),), {'__module__': 'vllm.v1.worker.gpu.block_table'})
        states.num_computed_tokens = b.StagedWriteTensor(8, np.int32, 'cuda:0')
        tables.block_tables = [b.StagedWriteTensor((8, 6), np.int32, 'cuda:0') for _ in range(3)]
        cls = nu.NativeUvaSlots
        with mock.patch.object(lc.lane_chunk.PinnedPorts, 'load', return_value=lc.fake_ports(calls)), \
             mock.patch.object(nu, 'NativeUvaSlots', side_effect=lambda runner, torch: cls(runner, d.torch, b)):
            f = lc.LaneChunkFactory(runner, torch_module=torch, prepare_ahead=True)
        self.assertIs(runner._flash_s5_native_uva, f.native_uva)
        with self.assertRaises(nu.NativeUvaError):
            f.admit(lc.admission())
        self.assertEqual(states.req_id_to_index, {})
        with d.scope(d.native), f.native_uva.scope():
            chunk = f.admit(lc.admission())
        for publish in (lambda: f.request.update(lc.admission()), lambda: chunk.finish_state()):
            with self.assertRaises(nu.NativeUvaError):
                publish()

    def test_real_legacy_adopted_prepare_laps_under_nt_and_na_without_fix(self):
        for nosync in (False, True):
            with self.subTest(nosync=nosync):
                native_value, copies = self.legacy_adopt(nosync, fixed=False)
                self.assertEqual(native_value, 0, 'old native update read a decode request index')
                self.assertEqual(copies, 0, 'adoption never ran an input tensor copy')


    def test_real_legacy_adopted_prepare_and_finalize_keep_native_and_stock_slots_private(self):
        for nosync in (False, True):
            with self.subTest(nosync=nosync):
                native_value, copies = self.legacy_adopt(nosync, fixed=True)
                self.assertEqual(native_value, 8)
                self.assertEqual(copies, 0)


    def legacy_adopt(self, nosync, *, fixed):
        # Real factory/adopt/_finish/LaneRequest.update; same stream simulator as na's own test.
        saved = dict(lc.lane_chunk.SLOTS_FAST)
        self.addCleanup(lambda: lc.lane_chunk.SLOTS_FAST.update(saved))
        lc.lane_chunk.SLOTS_FAST.update(selected=False)
        r = ahead.Rig()
        f = r.factory
        f.complete_chunk(r.first, consumers_complete=True)
        r.stage(nosync)
        r.dev.run_all()
        d = Device()
        d.linked, d.native, d.decode = r.dev, r.dev.prefill, r.dev.default
        b = pinned_buffers(d, depth=3)
        states = f.runner.req_states
        states.num_computed_tokens = b.StagedWriteTensor(8, np.int32, 'cuda:0')
        states.apply_staged_writes = states.num_computed_tokens.apply_write
        # Container/allocation leaves; actual pool and writer algorithms stay unchanged.
        states.__class__ = type('PinnedRequestState', (type(states),), {'__module__': 'vllm.v1.worker.gpu.states'})
        tables = f.runner.block_tables
        tables.__class__ = type('PinnedBlockTables', (type(tables),), {'__module__': 'vllm.v1.worker.gpu.block_table'})
        tables.block_tables = [b.StagedWriteTensor((8, 6), np.int32, 'cuda:0') for _ in range(3)]
        f.native_uva = nu.NativeUvaSlots(f.runner, d.torch, b) if fixed else None
        dispatcher = ld.LaneDispatcher.__new__(ld.LaneDispatcher)
        dispatcher.factory, dispatcher.torch = f, r.torch
        dispatcher.native = types.SimpleNamespace(split=types.SimpleNamespace(prefill_stream=r.dev.prefill))
        r.dev.h2d.clear()
        with dispatcher._scope():
            chunk = f.adopt(pa.later())
        copies = len(r.dev.h2d)
        # Two ordinary updates before PREPARE's queued descriptor reader runs.
        for index in (1, 0, 1):
            states.num_computed_tokens.stage_write_elem(index, 100 + index)
            states.apply_staged_writes()
        r.dev.run_stream(r.dev.prefill)
        value = int(states.num_computed_tokens.gpu[chunk.req_index])
        if fixed:
            r.dev.run_stream(r.dev.default)  # Retire stock's own three-batch window before its next window.
            # STATE_FINALIZE is another real shared-state publication, also scoped.
            with dispatcher._scope():
                chunk.finish_state()
            for index in (1, 0, 1):
                states.num_computed_tokens.stage_write_elem(index, 200 + index)
                states.apply_staged_writes()
            r.dev.run_stream(r.dev.prefill)
            self.assertEqual(int(states.num_computed_tokens.gpu[chunk.req_index]), chunk.admission.chunk_end)
        return value, copies


