"""Real pinned pool/writer algorithms, delayed CPU kernel/stream leaves; no CUDA."""
import ast
import contextlib
import functools
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
import native_uva as nu


def enqueue(stream, op):
    stream.ops.append(('uva-kernel-or-event', op) if hasattr(stream, 'name') else op)


def run_one(stream):
    op = stream.ops.pop(0)
    (op[1] if isinstance(op, tuple) else op)()


class Tensor(np.ndarray):
    def numpy(self):
        return self

    def numel(self):
        return self.size

    def element_size(self):
        return self.itemsize

    def stride(self, dim):
        return self.strides[dim] // self.itemsize

    def data_ptr(self):
        return self.ctypes.data


class UvaBuffer:
    def __init__(self, size, dtype):
        self.cpu = np.zeros(size, dtype=dtype).view(Tensor)
        self.np = self.uva = self.cpu


class Stream:
    def __init__(self):
        self.ops = []

    def run(self):
        while self.ops:
            self.ops.pop(0)()


class Event:
    def __init__(self, device):
        self.device, self.done, self.recorded = device, False, False

    def record(self):
        self.recorded = True
        self.stream = self.device.current
        enqueue(self.stream, lambda: setattr(self, 'done', True))

    def query(self):
        return self.done if self.recorded else True

    def synchronize(self):
        if not self.recorded:
            return
        self.device.waits += 1
        while not self.done:
            if not self.stream.ops:
                raise AssertionError('unrecorded completion')
            run_one(self.stream)


class Device:
    def __init__(self):
        self.native, self.decode = Stream(), Stream()
        self.current = self.decode
        self.waits = 0
        self.linked = None
        self.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            Event=lambda **kw: Event(self), stream=self.scope))

    @property
    def current(self):
        return self.linked.current[-1] if getattr(self, 'linked', None) else self._current

    @current.setter
    def current(self, stream):
        self._current = stream

    @contextlib.contextmanager
    def scope(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


def pinned_buffers(device, depth=2):
    """Compile the exact four real classes, replacing only allocation and launch."""
    path = ROOT / 'reference/vllm-src/v1/worker/gpu/buffer_utils.py'
    source = ast.parse(path.read_text())
    names = ('UvaBufferPool', 'UvaBackedTensor', 'StagedWriteTensor', 'FusedStagedWriter')
    classes = [n for n in source.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert len(classes) == 4

    class Kernel:
        def __getitem__(self, grid):
            def launch(output, stride, indices, starts, contents, ends, groups, **kw):
                def run():
                    last = 0
                    for i, index in enumerate(indices):
                        target = output[int(groups[i])] if groups is not None else output
                        end = int(ends[i])
                        flat = target.reshape(-1)
                        begin = int(index) * (target.stride(0)) + int(starts[i])
                        flat[begin:begin + end - last] = contents[last:end]
                        last = end
                enqueue(device.current, run)
            return launch

    torch = types.SimpleNamespace(Tensor=Tensor, int32=np.int32, int64=np.int64, float32=np.float32,
                                 zeros=lambda size, dtype, **kw: np.zeros(size, dtype=dtype).view(Tensor))
    namespace = dict(torch=torch, np=np, UvaBuffer=UvaBuffer, _DEFAULT_MAX_CONCURRENCY=depth,
                     partial=functools.partial, _apply_write_kernel=Kernel(),
                     async_tensor_h2d=lambda x, **kw: np.array(x, dtype=kw['dtype']).view(Tensor))
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + classes,
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def container(module, **values):
    cls = type('State', (types.SimpleNamespace,), {'__module__': module})
    return cls(**values)


def rig():
    device = Device()
    b = pinned_buffers(device)
    state = b.StagedWriteTensor(8, np.int32, 'cuda:0')
    scalar = b.UvaBackedTensor(8, np.int32)
    table = b.StagedWriteTensor((8, 8), np.int32, 'cuda:0')
    fused = b.FusedStagedWriter('cuda:0', 8)
    runner = types.SimpleNamespace(
        req_states=container('vllm.v1.worker.gpu.states', num_computed_tokens=state, prompt_len=scalar),
        block_tables=container('vllm.v1.worker.gpu.block_table', block_tables=[table], fused_writer=fused),
        model_state=types.SimpleNamespace(rope_state=None, prompt_embeds_state=None))
    owner = nu.NativeUvaSlots(runner, device.torch, b)
    return device, b, runner, owner


class Retirement(unittest.TestCase):
    def test_blocking_prepare_does_not_retire_preexisting_decode_readers(self):
        d = Device()
        b = pinned_buffers(d, depth=3)  # PP2 + async V2 -> three concurrent ordinary batches.
        state = b.StagedWriteTensor(8, np.int32, 'cuda:0')
        for i in range(3):
            state.stage_write_elem(i, 100 + i)
            state.apply_write()
        with d.scope(d.native):
            state.stage_write_elem(3, 203)
            state.apply_write()
            d.native.run()  # Even a complete native PREPARE cannot undo its earlier CPU overwrite.
        d.decode.run()
        self.assertEqual(int(state.gpu[0]), 0)
        self.assertEqual(int(state.gpu[3]), 100)




    def test_old_pool_reuses_a_live_decode_descriptor(self):
        d, b, runner, owner = rig()
        state = runner.req_states.num_computed_tokens
        state.stage_write_elem(1, 101)
        state.apply_write()  # Ordinary step; kernel has not read slot 1 yet.
        with d.scope(d.native):
            state.stage_write_elem(2, 202)
            state.apply_write()
            state.stage_write_elem(3, 303)
            state.apply_write()  # Native publications lap the depth-2 pool.
        d.decode.run()
        self.assertNotEqual(int(state.gpu[1]), 101)
        self.assertEqual(int(state.gpu[3]), 101, 'delayed decode used the native request index')

    def test_private_native_slots_preserve_both_stream_orders_and_stock_cadence(self):
        for order in ('native-first', 'decode-first'):
            with self.subTest(order=order):
                d, b, runner, owner = rig()
                state = runner.req_states.num_computed_tokens
                stock = state.write_indices
                state.stage_write_elem(1, 101)
                state.apply_write()
                curr = stock._curr
                for index, value in ((2, 202), (3, 303)):
                    with d.scope(d.native), owner.scope():
                        state.stage_write_elem(index, value)
                        state.apply_write()
                    self.assertIs(state.write_indices, stock)
                    self.assertEqual(stock._curr, curr)
                (d.native if order == 'native-first' else d.decode).run()
                (d.decode if order == 'native-first' else d.native).run()
                self.assertEqual(state.gpu.tolist()[:4], [0, 101, 202, 303])

    def test_retire_before_reuse_and_growth_on_third_native_publication(self):
        d, b, runner, owner = rig()
        state = runner.req_states.num_computed_tokens
        for index in range(1, 5):
            with d.scope(d.native), owner.scope():
                state.stage_write_elem(index, index * 11)
                state.apply_write()
        self.assertGreater(d.waits, 0)
        d.native.run()
        self.assertEqual(state.gpu.tolist()[:4], [0, 11, 22, 33])
        private = owner.pools[id(state.write_indices)]
        with d.scope(d.native), owner.scope():
            private.copy_to_uva(list(range(20)))
        with d.scope(d.native), owner.scope():
            private.copy_to_uva(list(range(40)))
        with d.scope(d.native), owner.scope():
            private.copy_to_uva(list(range(80)))
        self.assertEqual(private.pool._uva_bufs[private.pool._curr].cpu.shape[0], 128)

    def test_scalar_views_and_graph_addresses_are_restored_and_readers_retired(self):
        d, b, runner, owner = rig()
        scalar = runner.req_states.prompt_len
        stock_gpu, stock_pool, curr = scalar.gpu, scalar.pool, scalar.pool._curr
        observed = []
        scalar.np[2] = 22
        with d.scope(d.native), owner.scope():
            native_gpu = scalar.copy_to_uva()
        self.assertIs(scalar.gpu, stock_gpu)
        for _ in range(3):
            scalar.copy_to_uva()
        with d.scope(d.native), owner.scope():
            self.assertIs(scalar.gpu, native_gpu)
            d.native.ops.append(lambda view=scalar.gpu: observed.append(int(view[2])))
        with d.scope(d.native), owner.scope():
            scalar.np[2] = 33
            scalar.copy_to_uva()
        with d.scope(d.native), owner.scope():
            scalar.np[2] = 44
            scalar.copy_to_uva()
        with d.scope(d.native), owner.scope():
            scalar.np[2] = 55
            scalar.copy_to_uva()
        self.assertEqual(observed, [22])
        self.assertIs(scalar.pool, stock_pool)
        # Native did not alter the ordinary graph's captured allocation.
        self.assertTrue(np.shares_memory(stock_gpu, stock_pool._uva_bufs[curr].cpu))

    def test_fused_writer_uses_the_same_isolation(self):
        d, b, runner, owner = rig()
        table, fused = runner.block_tables.block_tables[0], runner.block_tables.fused_writer
        table.stage_write(1, 0, [11, 12])
        fused.apply([table], [table.gpu], [8])
        for index in (2, 3):
            with d.scope(d.native), owner.scope():
                table.stage_write(index, 0, [index * 11, index * 11 + 1])
                fused.apply([table], [table.gpu], [8])
        d.decode.run()
        d.native.run()
        self.assertEqual(table.gpu[:4, :2].tolist(), [[0, 0], [11, 12], [22, 23], [33, 34]])

    def test_body_and_record_failures_restore_stock_and_never_reuse_unproved_slots(self):
        for record_error in (False, True):
            d, b, runner, owner = rig()
            state = runner.req_states.num_computed_tokens
            stock = state.write_indices
            if record_error:
                d.torch.cuda.Event = mock.Mock(side_effect=OSError('event creation'))
            with self.assertRaises((OSError, ValueError)):
                with d.scope(d.native), owner.scope():
                    state.stage_write_elem(2, 22)
                    state.apply_write()
                    if not record_error:
                        raise ValueError('after enqueue')
            self.assertIs(state.write_indices, stock)
            self.assertFalse(owner.active)
            if record_error:
                self.assertTrue(owner.failed)
                with self.assertRaises(nu.NativeUvaError):
                    with owner.scope():
                        pass
            else:
                d.native.run()
                self.assertEqual(state.gpu[2], 22)

    def test_unfinished_segment_cannot_lap_itself(self):
        d, b, runner, owner = rig()
        state = runner.req_states.num_computed_tokens
        with self.assertRaises(nu.NativeUvaError):
            with d.scope(d.native), owner.scope():
                for i in range(4):
                    state.stage_write_elem(i, i)
                    state.apply_write()

    def test_unclassified_pool_and_foreign_binding_refuse(self):
        d, b, runner, owner = rig()
        state = runner.req_states.num_computed_tokens
        state.write_indices = object()
        with self.assertRaises(nu.NativeUvaError):
            with owner.scope():
                pass
        with self.assertRaises(nu.NativeUvaError):
            nu.NativeUvaSlots(runner, d.torch, b)


if __name__ == '__main__':
    unittest.main()
