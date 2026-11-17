"""CPU-only tests for lever 2's soft split (FLASH_S5_DECODE_OVERLAP_SMS, patches/s5_engine/decode_overlap.py).

Fakes: the green helper binding, torch and the stock cudagraph manager of test_e3_green_native / test_e3_native_startup
(no torch, no vLLM, no CUDA). They exercise the flag parse, the wide pair's measurement and refusals, the overlap view's
delegation, and the real install_worker_native / split bank / route controller / shutdown composition with the view.
They cannot establish that the driver accepts an overlapping green context, where decode blocks land or what it costs.
"""
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
sys.path.insert(0, str(ROOT / 'harness'))

import decode_overlap  # noqa: E402
from decode_overlap import DecodeOverlapError, OverlapDecodeSplit, declared, install, open_wide  # noqa: E402
from green_context import GreenSplit  # noqa: E402
import split_bank  # noqa: E402
from test_e3_green_native import FakeBinding, FakeTensor, fake_torch  # noqa: E402
import test_e3_native_startup as startup  # noqa: E402

FLAG = {'FLASH_S5_DECODE_OVERLAP_SMS': '42'}


class Memory:
    """torch.cuda.mem_get_info for the fake torch: free drops by ``step`` at every call after ``arm``."""

    def __init__(self, torch, step=0):
        self.free = 5_000_000_000
        self.step = step
        self.calls = 0
        torch.cuda.mem_get_info = self

    def __call__(self, index=None):
        self.calls += 1
        free = self.free - (self.step if self.calls > 1 else 0)
        return (free, 24_000_000_000)


class ShiftedBinding(FakeBinding):
    """A wide decode group placed at ``offset`` (the fake's default is the first N SM ids)."""

    def __init__(self, offset, **kw):
        super().__init__(**kw)
        self.offset = offset

    def probe(self, pointer, blocks, stream_token, cycles):
        role, dc = self.streams[stream_token]
        if role == 'decode' and dc > 24:
            tensor = FakeTensor.registry[pointer]
            sms = [(self.offset + i) % self.total for i in range(dc)]
            for i in range(blocks):
                tensor.values[i] = sms[i % len(sms)]
            return 0
        if role == 'prefill' and dc > 24:
            tensor = FakeTensor.registry[pointer]
            used = {(self.offset + i) % self.total for i in range(dc)}
            sms = [s for s in range(self.total) if s not in used]
            for i in range(blocks):
                tensor.values[i] = sms[i % len(sms)]
            return 0
        return super().probe(pointer, blocks, stream_token, cycles)


def base_split(torch, binding=None):
    split = GreenSplit(binding or FakeBinding(), 0, 24, torch_module=torch).create()
    assert split.membership().ok
    return split


class Flag(unittest.TestCase):
    def test_off_values(self):
        self.assertIsNone(declared({}))
        self.assertIsNone(declared({'FLASH_S5_DECODE_OVERLAP_SMS': '0'}))

    def test_on_value(self):
        self.assertEqual(declared(FLAG), 42)

    def test_refused_values(self):
        for value in ('41', 'x', '-42', '042', '', ' 42', '4.2'):
            with self.assertRaises(DecodeOverlapError, msg=value):
                declared({'FLASH_S5_DECODE_OVERLAP_SMS': value})


class WidePair(unittest.TestCase):
    def setUp(self):
        FakeTensor.registry.clear()
        self.torch = fake_torch(82)
        self.memory = Memory(self.torch)

    def test_off_returns_the_base_object_and_creates_nothing(self):
        base = base_split(self.torch)
        created = len(base.binding.created)
        self.assertIs(install(base, {}), base)
        self.assertIs(install(base, {'FLASH_S5_DECODE_OVERLAP_SMS': '0'}), base)
        self.assertEqual(len(base.binding.created), created)
        self.assertEqual(self.memory.calls, 0)

    def test_on_measures_and_records_the_overlap(self):
        base = base_split(self.torch)
        lines = []
        view = install(base, FLAG, log=lines.append)
        self.assertIsInstance(view, OverlapDecodeSplit)
        self.assertEqual(len(base.binding.created), 2)
        rec = view.overlap
        self.assertEqual(lines, [rec])
        self.assertEqual((rec['requested_sms'], rec['decode_sms'], rec['own_decode_sms'], rec['borrowed_prefill_sms']),
                         (42, 42, 24, 18))
        self.assertIs(rec['superset'], True)
        self.assertEqual((rec['base_decode_sms'], rec['base_prefill_sms'], rec['wide_prefill_sms_unused']), (24, 58, 40))
        self.assertEqual(rec['pair_delta_bytes'], 0)
        self.assertTrue(rec['membership']['disjoint'])
        self.assertEqual(rec['schema'], 's5-decode-overlap/1')

    def test_view_delegates_decode_to_wide_and_the_rest_to_base(self):
        base = base_split(self.torch)
        view = install(base, FLAG, log=lambda r: None)
        wide = view.wide
        for name in ('decode_stream', 'decode_stream_token', 'decode_context_id'):
            self.assertIs(getattr(view, name), getattr(wide, name), name)
            self.assertIsNot(getattr(view, name), getattr(base, name), name)
        self.assertEqual((view.decode_sms, view.requested_decode_sms), (42, 42))
        for name in ('prefill_stream', 'prefill_stream_token', 'prefill_context_id', 'binding', 'device_index',
                     'membership_result', 'torch'):
            self.assertIs(getattr(view, name), getattr(base, name), name)
        self.assertEqual((view.prefill_sms, view.total_sms), (58, 82))
        self.assertEqual(view.state, 'open')
        # the startup grade's split record stays the base pair's; the overlap rides beside it
        record = view.describe()
        self.assertEqual((record['requested_decode_sms'], record['decode_sms'], record['prefill_sms'],
                          record['total_sms']), (24, 24, 58, 82))
        self.assertEqual(record['membership']['decode_seen'], 24)
        self.assertEqual(record['decode_overlap'], view.overlap)
        self.assertEqual({k: v for k, v in record.items() if k != 'decode_overlap'}, base.describe())
        with self.assertRaises(AttributeError):
            view.no_such_attribute
        with self.assertRaises(DecodeOverlapError):
            view.membership()

    def test_aux_streams_go_to_their_own_context(self):
        base = base_split(self.torch)
        view = install(base, FLAG, log=lambda r: None)
        decode_aux = view.create_decode_aux()
        self.assertIn(decode_aux.cuda_stream, view.wide._aux_streams)
        self.assertNotIn(decode_aux.cuda_stream, base._aux_streams)
        prefill_aux = view.create_prefill_aux()
        self.assertIn(prefill_aux.cuda_stream, base._prefill_aux_streams)
        self.assertNotIn(prefill_aux.cuda_stream, view.wide._prefill_aux_streams)
        view.check_decode_stream(view.decode_stream)
        view.check_prefill_stream(prefill_aux)
        view.destroy_decode_aux(decode_aux)
        self.assertNotIn(decode_aux.cuda_stream, view.wide._aux_streams)

    def test_graph_owner_retains_both_allocator_contexts(self):
        base = base_split(self.torch)
        view = install(base, FLAG, log=lambda r: None)
        owner = object()
        view.retain_graph_owner(owner)
        self.assertTrue(base.lifetime_record()['allocator_exposed'])
        self.assertTrue(view.wide.lifetime_record()['allocator_exposed'])
        self.assertTrue(view.lifetime_record()['allocator_exposed'])
        self.assertIn('decode_overlap', view.lifetime_record())
        with self.assertRaises(Exception):
            view.close()            # the wide pair is still owned by graphs
        view.release_graph_owner(owner)

    def test_close_wide_then_base(self):
        base = base_split(self.torch)
        view = install(base, FLAG, log=lambda r: None)
        handles = list(base.binding.created)
        self.assertTrue(view.close())
        self.assertEqual(base.binding.closed, [handles[1], handles[0]])
        self.assertEqual((view.wide.state, base.state, view.state), ('closed', 'closed', 'closed'))

    def test_count_out_of_range_refused_before_any_create(self):
        base = base_split(self.torch)
        for value in ('24', '22', '82', '84'):
            with self.assertRaises(DecodeOverlapError, msg=value):
                install(base, {'FLASH_S5_DECODE_OVERLAP_SMS': value}, log=lambda r: None)
        self.assertEqual(len(base.binding.created), 1)

    def test_unmeasured_base_refused(self):
        base = GreenSplit(FakeBinding(), 0, 24, torch_module=self.torch).create()
        with self.assertRaisesRegex(DecodeOverlapError, 'membership'):
            install(base, FLAG, log=lambda r: None)
        base.close()
        with self.assertRaisesRegex(DecodeOverlapError, 'open base'):
            install(base, FLAG, log=lambda r: None)

    def test_wide_membership_failure_refused_and_wide_closed(self):
        base = base_split(self.torch)
        base.binding.overlap = True       # the fake's prefill probe now lands on a decode SM
        with self.assertRaisesRegex(DecodeOverlapError, 'membership'):
            install(base, FLAG, log=lambda r: None)
        self.assertEqual(base.binding.closed, [base.binding.created[1]])
        self.assertEqual(base.state, 'open')

    def test_memory_drop_above_limit_refused(self):
        self.memory.step = decode_overlap.MAX_PAIR_BYTES + 1
        base = base_split(self.torch)
        with self.assertRaisesRegex(DecodeOverlapError, 'device memory'):
            install(base, FLAG, log=lambda r: None)
        self.assertEqual(base.binding.closed, [base.binding.created[1]])

    def test_memory_drop_at_limit_recorded(self):
        self.memory.step = decode_overlap.MAX_PAIR_BYTES
        view = install(base_split(self.torch), FLAG, log=lambda r: None)
        self.assertEqual(view.overlap['pair_delta_bytes'], decode_overlap.MAX_PAIR_BYTES)

    def test_wide_group_without_decode_sms_refused(self):
        base = base_split(self.torch, ShiftedBinding(24))   # wide = SMs 24..65: prefill only
        with self.assertRaisesRegex(DecodeOverlapError, 'holds 0 decode'):
            install(base, FLAG, log=lambda r: None)
        self.assertEqual(base.binding.closed, [base.binding.created[1]])

    def test_partial_superset_recorded_not_refused(self):
        base = base_split(self.torch, ShiftedBinding(2))    # wide = SMs 2..43: 22 decode + 20 prefill
        view = install(base, FLAG, log=lambda r: None)
        self.assertEqual((view.overlap['own_decode_sms'], view.overlap['borrowed_prefill_sms']), (22, 20))
        self.assertIs(view.overlap['superset'], False)

    def test_driver_count_mismatch_refused(self):
        class Rounding(FakeBinding):
            def create(self, device, requested):
                return super().create(device, requested + 2 if requested > 24 else requested)
        base = base_split(self.torch, Rounding())
        with self.assertRaisesRegex(DecodeOverlapError, 'declared 42'):
            install(base, FLAG, log=lambda r: None)
        self.assertEqual(base.binding.closed, [base.binding.created[1]])


class Composition(unittest.TestCase):
    """The real install_worker_native, split bank, route controller and shutdown over the overlap view."""

    def setUp(self):
        startup.StartupTests.setUp(self)
        self.memory = Memory(self.torch)
        self.split = GreenSplit(FakeBinding(), 0, 24, torch_module=self.torch).create()
        self.assertTrue(self.split.membership().ok)
        self.runner._flash_s5_split = self.split
        self.runner._flash_s5_native_identity = self.split.describe()

    def tearDown(self):
        startup.StartupTests.tearDown(self)

    def install(self, env):
        return startup.install_worker_native(self.runner, chunk_tokens=1024, torch_module=self.torch,
                                             pynccl_class=startup.FakeNccl, tp_group=self.tp, pp_group=self.pp,
                                             env=env)

    def test_off_keeps_the_base_split_everywhere(self):
        native = self.install({'FLASH_S5_DECODE_OVERLAP_SMS': '0'})
        self.assertIs(native.split, self.split)
        self.assertIs(native.bank.split, self.split)
        self.assertIs(self.runner._flash_s5_split, self.split)
        self.assertNotIn('decode_overlap', native.inventory()['split'])
        self.assertEqual(len(self.split.binding.created), 1)

    def test_on_bank_and_route_on_the_wide_context_native_on_the_base_prefill(self):
        native = self.install(FLAG)
        view = native.split
        self.assertIsInstance(view, OverlapDecodeSplit)
        self.assertIs(view.base, self.split)
        self.assertIs(native.bank.split, view)
        self.assertIs(self.runner._flash_s5_split, view)
        # the identity row the startup grade reads at worker init is untouched
        self.assertEqual(self.runner._flash_s5_native_identity['decode_sms'], 24)
        inventory = native.inventory()
        self.assertEqual((inventory['split']['requested_decode_sms'], inventory['split']['decode_sms'],
                          inventory['split']['prefill_sms'], inventory['split']['total_sms']), (24, 24, 58, 82))
        self.assertEqual(inventory['split']['membership']['decode_seen'], 24)
        self.assertEqual(inventory['split']['decode_overlap']['borrowed_prefill_sms'], 18)
        # decode storage and the bank's execution context name the wide decode context
        self.assertEqual(native.decode_owner.context_id, view.wide.decode_context_id)
        context = native.bank.execution_context()
        self.assertEqual((context.context_id, context.stream_token, context.measured_sm, context.requested_sm),
                         (view.wide.decode_context_id, view.wide.decode_stream_token, 42, 42))
        # the native lane's streams are the base pair's prefill context
        self.assertIs(native.split.prefill_stream, self.split.prefill_stream)
        self.assertEqual(native.prefill_streams.record()['basis'], 'prefill-green')
        self.assertIs(native.ple_scratch.stream, native.prefill_streams.replacements[0])
        self.assertIn(native.ple_scratch.stream.cuda_stream, self.split._prefill_aux_streams)
        with split_bank.prefill_scope(self.runner) as stream:
            self.assertIs(stream, self.split.prefill_stream)
        # capture on the wide stream, then the drained route switch makes it the runner's main stream
        native.bank.capture()
        self.assertIn(native.bank.capture_streams.split, (view,))
        for stream in native.bank.capture_streams.streams:
            self.assertIn(stream.cuda_stream, view.wide._aux_streams)
        ack = startup.transition_rpc(self.runner, dict(engine_epoch=1, expected_route_generation=0,
                                                       route_generation=1, target='decode-split',
                                                       storage_epoch=None), engine_epoch=1)
        self.assertEqual(ack['route'], 'split')
        self.assertIs(self.runner.main_stream, view.wide.decode_stream)
        self.assertIsNot(self.runner.main_stream, self.split.decode_stream)
        with split_bank.decode_scope(self.runner):
            self.assertIs(self.torch.cuda.current_stream(), view.wide.decode_stream)
        startup.transition_rpc(self.runner, dict(engine_epoch=1, expected_route_generation=1, route_generation=2,
                                                 target='full'), engine_epoch=1)
        self.assertIsNone(self.runner._flash_s5_decode_stream)
        # the decode timeline's bank_sms reads runner._flash_s5_split: the wide count on the split bank
        self.assertEqual((self.runner._flash_s5_split.decode_sms, self.runner._flash_s5_split.total_sms), (42, 82))

    def test_on_shutdown_closes_wide_then_base(self):
        native = self.install(FLAG)
        native.bank.capture()
        handles = list(self.split.binding.created)
        self.assertTrue(split_bank.shutdown(self.runner))
        self.assertEqual(native.split.wide.state, 'closed')
        self.assertEqual(self.split.state, 'closed')
        self.assertEqual(handles[0] in self.split.binding.closed or self.split.native_retained, True)

    def test_on_refusal_stops_startup_before_the_bank(self):
        self.split.binding.overlap = True
        with self.assertRaises(DecodeOverlapError):
            self.install(FLAG)
        self.assertIsNone(getattr(self.runner, '_flash_s5_split_bank', None))
        self.assertIsNone(getattr(self.runner, '_flash_s5_native', None))


class OffPath(unittest.TestCase):
    def test_off_never_imports_the_module(self):
        code = ('import sys; sys.path[:0] = [%r, %r]; import native_startup; '
                'assert "decode_overlap" not in sys.modules; print("ok")'
                % (str(ROOT / 'patches/s5_engine'), str(ROOT / 'harness')))
        out = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True, timeout=120)
        self.assertEqual((out.returncode, out.stdout.strip()), (0, 'ok'), out.stderr[-2000:])

    def test_native_startup_hook_is_the_only_entry(self):
        text = (ROOT / 'patches/s5_engine/native_startup.py').read_text()
        self.assertEqual(text.count('decode_overlap'), 3)   # the import line twice, the call once
        self.assertEqual(text.count("get('FLASH_S5_DECODE_OVERLAP_SMS', '0') != '0'"), 1)
        idx = text.index('install_decode_overlap(split')
        self.assertLess(idx, text.index('bank, backend, controller = install_split_bank(\n        runner, split,'))
        # the hook sits before the bank of the worker-native startup, not in the decode-probe startup
        self.assertLess(idx, text.index('def install_decode_probe'))


if __name__ == '__main__':
    unittest.main()
