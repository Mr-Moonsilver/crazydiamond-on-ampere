#!/usr/bin/env python3
"""parity-ahead (tmp/decode-phase/READING-parity-gaps-20261107.md A.5): the staged successor's copies without a synchronize.

Hub, no GPU. A fake device with streams and events: an op runs only when the test runs it, in a stream order the test
chooses, a blocking host-to-device copy drains its stream first (PyTorch's memcpy_and_sync), and a DMA reads its pinned host
buffer when it executes, through a weak reference (a dropped buffer reads FREED). Under it run the real LaneChunkFactory and
LaneTables (test_e3_lane_chunk's runner and ports), stage_wire's worker path, the engine's _ScopedSender and the overlap
driver's selection.

What this proves:
- off: the ahead's first device tensor waits for the prefill stream (the live chunk's queued tail runs inside the call);
- on: nothing waits; every host value reaches the device by a pinned, non-blocking copy on the prefill stream; the later
  readers on that stream see exactly the off path's values in either stream order; the live chunk's readers queued before the
  ahead still read their own table row; the pinned sources live until the event recorded after the last copy has completed,
  and go after;
- the engine wraps the record only when the arm selects it, the worker stages a wrapped record with nosync and a plain one as
  before (no keyword arguments at all with the receipt not installed), the receipt line and its fields;
- the selection's refusals, echo and status, and the control client's checks.
What it does not: real CUDA, the stock builders' own host-device behaviour inside build_metadata (the box receipt's busy_out
reads that), timing.
"""
import asyncio
import contextlib
import gc
import os
import pickle
import types
import unittest
import weakref
from unittest import mock

import test_e3_lane_chunk as tlc
import test_prepare_ahead as pa
import lane_chunk as lc
import stage_wire as sw

POISON = 'POISON'       # a device element no op has written yet
FREED = 'FREED'         # a DMA read from a host buffer that was dropped before the copy ran


# --- the fake device ---------------------------------------------------------------------------------------------------------

class Stream:
    def __init__(self, device, name):
        self.device, self.name, self.ops = device, name, []

    def query(self):
        return not self.ops

    def synchronize(self):
        self.device.sync(self, 'stream-synchronize')

    def __repr__(self):
        return 'Stream(%s)' % self.name


class Event:
    def __init__(self, device, enable_timing=False):
        self.device, self.recorded, self.done = device, False, False

    def record(self, stream=None):
        stream = stream or self.device.current[-1]
        self.recorded, self.done = True, False
        stream.ops.append(('event', lambda: setattr(self, 'done', True)))
        self.stream = stream.name

    def query(self):
        return self.done if self.recorded else True

    def synchronize(self):
        self.device.syncs.append(('event-synchronize', self.stream))


class Store:
    """One allocation; a weak reference to it is what a DMA holds."""

    def __init__(self, values, device, pinned=False):
        self.values, self.device, self.pinned = list(values), device, pinned


def _flat(values):
    values = list(values)
    if values and isinstance(values[0], (list, tuple)):
        rows = [list(r) for r in values]
        return [v for r in rows for v in r], (len(rows), len(rows[0]) if rows else 0)
    return values, (len(values),)


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


class DT:
    """A strided-free tensor over a Store: offset, shape, dtype."""

    def __init__(self, device, store, shape, dtype, offset=0):
        self.dev, self.store, self.shape, self.dtype, self.offset = device, store, tuple(shape), dtype, offset

    @property
    def device(self):
        return self.store.device

    @property
    def values(self):
        return self.store.values[self.offset:self.offset + self.numel()]

    def numel(self):
        return _numel(self.shape)

    def element_size(self):
        return 8 if self.dtype == 'i64' else 4

    def data_ptr(self):
        return id(self.store) * 1000 + self.offset

    def is_pinned(self):
        return self.store.device == 'cpu' and self.store.pinned

    def pin_memory(self):
        if self.device != 'cpu':
            raise RuntimeError('pin_memory on a device tensor')
        return DT(self.dev, Store(self.values, 'cpu', True), self.shape, self.dtype)

    def tolist(self):
        if self.device != 'cpu':
            self.dev.sync(self.dev.current[-1], 'readback')
        return list(self.values)

    def view(self, *shape):
        return DT(self.dev, self.store, shape, self.dtype, self.offset)

    def clone(self):
        return DT(self.dev, Store(self.values, self.device, False), self.shape, self.dtype)

    # CPU-only elementwise arithmetic and gathers: what LaneTables.slot_mappings_fast does on the host (never on the device).
    def _host(self):
        if self.device != 'cpu':
            raise RuntimeError('host arithmetic on a device tensor')
        return self.values

    def _binary(self, other, op):
        a = self._host()
        b = other._host() if isinstance(other, DT) else [other] * len(a)
        return DT(self.dev, Store([op(x, y) for x, y in zip(a, b)], 'cpu'), self.shape, self.dtype)

    def __floordiv__(self, other):
        return self._binary(other, lambda x, y: x // y)

    def __mod__(self, other):
        return self._binary(other, lambda x, y: x % y)

    def __mul__(self, other):
        return self._binary(other, lambda x, y: x * y)

    def __add__(self, other):
        return self._binary(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._binary(other, lambda x, y: x - y)

    def __setitem__(self, item, value):
        if type(item) is not int or not isinstance(value, DT):
            raise TypeError('only a whole row from a tensor')
        self._host()
        width = self.numel() // self.shape[0]
        values = value._host()
        if len(values) != width:
            raise RuntimeError('row sizes differ')
        self.store.values[self.offset + item * width:self.offset + (item + 1) * width] = list(values)

    def __getitem__(self, item):
        if isinstance(item, DT):
            values = self._host()
            return DT(self.dev, Store([values[i] for i in item._host()], 'cpu'), item.shape, self.dtype)
        if isinstance(item, tuple):
            row, cols = item
            width = self.shape[1]
            lo, hi, _ = cols.indices(width)
            return DT(self.dev, self.store, (hi - lo,), self.dtype, self.offset + row * width + lo)
        if isinstance(item, slice):
            lo, hi, _ = item.indices(self.shape[0])
            width = self.numel() // self.shape[0] if self.shape[0] else 0
            return DT(self.dev, self.store, (hi - lo,) + self.shape[1:], self.dtype, self.offset + lo * width)
        width = self.numel() // self.shape[0]
        return DT(self.dev, self.store, self.shape[1:] or (1,), self.dtype, self.offset + item * width)

    def _write(self, values):
        self.store.values[self.offset:self.offset + len(values)] = list(values)

    def copy_(self, src, non_blocking=False):
        if self.numel() != src.numel():
            raise RuntimeError('copy sizes differ')
        dev = self.dev
        if self.device == 'cpu' and src.device == 'cpu':
            self._write(src.values)
            return self
        if self.device != 'cpu' and src.device == 'cpu':
            stream = dev.current[-1]
            dev.h2d.append((stream.name, src.is_pinned(), bool(non_blocking)))
            if non_blocking and src.is_pinned():
                ref, n, start = weakref.ref(src.store), src.numel(), src.offset

                def dma():
                    store = ref()
                    self._write([FREED] * n if store is None else store.values[start:start + n])
                stream.ops.append(('h2d', dma))
                return self
            if non_blocking:
                # A pageable source: CUDA may stage it synchronously; the lane never relies on either answer.
                dev.syncs.append(('pageable-async-copy', stream.name))
            values = list(src.values)
            stream.ops.append(('h2d-blocking', lambda: self._write(values)))
            dev.sync(stream, 'blocking-copy')
            return self
        if src.device != 'cpu' and self.device != 'cpu':
            stream = dev.current[-1]
            stream.ops.append(('d2d', lambda: self._write(src.values)))
            return self
        dev.sync(dev.current[-1], 'device-to-host')
        self._write(src.values)
        return self

    def _fill(self, value):
        if self.device == 'cpu':
            self._write([value] * self.numel())
        else:
            n = self.numel()
            self.dev.current[-1].ops.append(('fill', lambda: self._write([value] * n)))
        return self

    def zero_(self):
        return self._fill(0)

    def fill_(self, value):
        return self._fill(value)

    def to(self, device, non_blocking=False):
        if device == self.device:
            return self
        out = self.dev.empty(self.shape, dtype=self.dtype, device=device)
        return out.copy_(self, non_blocking=non_blocking)


class Device:
    def __init__(self):
        self.default, self.prefill, self.side = (Stream(self, n) for n in ('default', 'prefill', 'side'))
        self.current = [self.default]
        self.syncs, self.h2d, self.done = [], [], []

    def stream(self, name):
        return dict(default=self.default, prefill=self.prefill, side=self.side)[name]

    def run_stream(self, stream, count=None):
        while stream.ops and (count is None or count > 0):
            label, op = stream.ops.pop(0)
            op()
            self.done.append((stream.name, label))
            if count is not None:
                count -= 1

    def sync(self, stream, reason):
        self.syncs.append((reason, stream.name))
        self.run_stream(stream)

    def run_all(self, order=('default', 'prefill', 'side')):
        for name in order:
            self.run_stream(self.stream(name))

    def empty(self, shape, dtype=None, device='cpu'):
        shape = (shape,) if type(shape) is int else tuple(shape)
        fill = 0 if device == 'cpu' else POISON
        return DT(self, Store([fill] * _numel(shape), device), shape, dtype)

    def torch(self):
        dev = self
        torch = types.SimpleNamespace(int32='i32', int64='i64', bool='bool', Tensor=(tlc.T, DT))

        def tensor(values, dtype=None, device='cpu', pin_memory=False):
            flat, shape = _flat(values)
            host = DT(dev, Store(flat, 'cpu', pin_memory), shape, dtype)
            if device == 'cpu':
                return host
            out = dev.empty(shape, dtype=dtype, device=device)
            return out.copy_(host, non_blocking=False)

        def zeros(shape, dtype=None, device='cpu'):
            return dev.empty(shape, dtype=dtype, device=device).zero_()

        def ones(shape, dtype=None, device='cpu'):
            return dev.empty(shape, dtype=dtype, device=device).fill_(1)

        def arange(lo, hi, dtype=None, device='cpu'):
            out = dev.empty((hi - lo,), dtype=dtype, device=device)
            if device == 'cpu':
                out._write(range(lo, hi))
            else:
                dev.current[-1].ops.append(('arange', lambda: out._write(range(lo, hi))))
            return out

        def full(shape, value, dtype=None):
            return dev.empty(shape, dtype=dtype).fill_(value)

        @contextlib.contextmanager
        def stream_scope(stream):
            dev.current.append(stream)
            try:
                yield stream
            finally:
                dev.current.pop()

        def synchronize():
            for s in (dev.default, dev.prefill, dev.side):
                dev.sync(s, 'device-synchronize')

        torch.tensor, torch.zeros, torch.ones, torch.arange, torch.full = tensor, zeros, ones, arange, full
        torch.empty = dev.empty
        torch.inference_mode = contextlib.nullcontext
        torch.cuda = types.SimpleNamespace(Event=lambda enable_timing=False: Event(dev, enable_timing),
                                           stream=stream_scope, current_stream=lambda: dev.current[-1],
                                           Stream=lambda *a, **kw: dev.side,
                                           synchronize=synchronize)
        return torch


# --- the rig -----------------------------------------------------------------------------------------------------------------

def vals(t):
    return list(t.values) if isinstance(t, DT) else t


def staged_values(chunk):
    out = dict(input_ids=chunk.input_ids, positions=chunk.positions, query_start_loc=chunk.query_start_loc,
               seq_lens=chunk.seq_lens, idx_mapping=chunk.idx_mapping, is_padding=chunk.is_padding,
               ngram_context=chunk.ngram_context, slot_mappings=chunk.slot_mappings,
               logits_indices=chunk.batch.logits_indices, cu_num_logits=chunk.batch.cu_num_logits,
               expanded_local_pos=chunk.batch.expanded_local_pos)
    out.update({'table%d' % g: view for g, view in enumerate(chunk.block_tables)})
    return {k: vals(v) for k, v in out.items()}


class Rig:
    """A factory with prepare-ahead, its first chunk admitted and live (holding set 0) and settled on the device."""

    def __init__(self):
        self.dev = Device()
        self.torch = self.dev.torch()
        self.calls = []
        self.factory = lc.LaneChunkFactory(tlc.fake_runner(self.torch, self.calls), ports=tlc.fake_ports(self.calls),
                                           torch_module=self.torch, prepare_ahead=True)
        self.first = self.factory.admit(tlc.admission())
        self.dev.run_all()
        self.seen = {}

    def live_tail(self):
        """n's queued tail on the prefill stream: one op, then n's own reader of its table rows (bound by view)."""
        rows = [view for view in self.first.block_tables]
        self.dev.prefill.ops.append(('tail-n', lambda: None))
        self.dev.prefill.ops.append(('n-reads-rows', lambda: self.seen.__setitem__('n_rows', [vals(v) for v in rows])))

    def stage(self, nosync, admission=None):
        self.dev.syncs.clear()
        self.dev.h2d.clear()
        with self.torch.cuda.stream(self.dev.prefill):
            if nosync:
                return self.factory.stage_ahead(admission or pa.later(), nosync=True)
            return self.factory.stage_ahead(admission or pa.later())

    def staged(self):
        return self.factory._staged[1]

    def reader(self):
        """n+1's later reader on the prefill stream (its first layer)."""
        chunk = self.staged()
        self.dev.prefill.ops.append(('n1-reads', lambda: self.seen.__setitem__('n1', staged_values(chunk))))


class Reset(pa.Reset):
    def setUp(self):
        super().setUp()
        saved = dict(lc.AHEAD_NOSYNC)
        self.addCleanup(lambda: lc.AHEAD_NOSYNC.update(saved))
        lc.AHEAD_NOSYNC.update(dict.fromkeys(lc.AHEAD_NOSYNC, 0))
        # The slot path is pinned per test (other suites in the same process select the fast one); SlotsFast runs it.
        slots = dict(lc.SLOTS_FAST)
        self.addCleanup(lambda: lc.SLOTS_FAST.update(slots))
        lc.SLOTS_FAST.update(selected=False, calls=0)
        tlc.FakeMambaContext.instances.clear()


def reference_values():
    """The off path's staged chunk, fully drained: what every reader must see."""
    rig = Rig()
    rig.stage(False)
    rig.dev.run_all()
    return staged_values(rig.staged())


# --- the worker's staging ----------------------------------------------------------------------------------------------------

class Off(Reset, unittest.TestCase):
    def test_the_off_path_waits_for_the_prefill_stream_inside_the_call(self):
        rig = Rig()
        rig.live_tail()
        self.assertEqual(rig.stage(False), dict(staged=True, why='staged'))
        self.assertIn(('blocking-copy', 'prefill'), rig.dev.syncs, 'the first device tensor drains the stream')
        self.assertIn(('prefill', 'tail-n'), rig.dev.done, "the live chunk's tail ran inside the ahead (the hold)")
        self.assertEqual(rig.seen['n_rows'][0][:4], [5, 6, 7, 0], "n's reader ran before the in-place write")
        self.assertIsNone(rig.factory.h2d_keep, 'nothing kept on the off path')
        self.assertEqual(lc.AHEAD_NOSYNC['staged'], 0)

    def test_the_off_call_passes_no_new_argument_to_the_chunk(self):
        rig = Rig()
        with mock.patch.object(lc, 'LaneChunk', wraps=lc.LaneChunk) as made:
            rig.stage(False)
        self.assertNotIn('staging', made.call_args.kwargs)
        self.assertNotIn('probe', made.call_args.kwargs)


class NoSync(Reset, unittest.TestCase):
    def test_nothing_waits_and_the_live_tail_is_still_queued_at_return(self):
        rig = Rig()
        rig.live_tail()
        result = rig.stage(True)
        self.assertEqual((result['staged'], result['why'], result['nosync']), (True, 'staged', True))
        self.assertEqual(rig.dev.syncs, [], 'no synchronize, blocking copy, readback or pageable async copy')
        self.assertFalse(rig.dev.prefill.query(), 'the prefill stream is still busy when the ahead returns')
        self.assertNotIn(('prefill', 'tail-n'), rig.dev.done)

    def test_every_host_value_is_a_pinned_non_blocking_copy_on_the_prefill_stream(self):
        rig = Rig()
        result = rig.stage(True)
        self.assertTrue(rig.dev.h2d)
        self.assertEqual(set(rig.dev.h2d), {('prefill', True, True)})
        self.assertEqual(result['copies'], len(rig.dev.h2d))
        self.assertGreater(result['bytes'], 0)
        self.assertEqual((lc.AHEAD_NOSYNC['staged'], lc.AHEAD_NOSYNC['copies']), (1, len(rig.dev.h2d)))

    def test_later_readers_see_the_off_paths_values_in_either_stream_order(self):
        want = reference_values()
        for order in (('prefill', 'default', 'side'), ('side', 'default', 'prefill')):
            with self.subTest(order=order):
                rig = Rig()
                rig.live_tail()
                rig.stage(True)
                rig.reader()
                gc.collect()
                rig.dev.run_all(order)
                got = rig.seen['n1']
                self.assertEqual(got, want)
                self.assertFalse([k for k, v in got.items() if POISON in v or FREED in v])

    def test_the_live_chunk_still_reads_its_own_rows(self):
        for order in (('prefill', 'default', 'side'), ('side', 'default', 'prefill')):
            with self.subTest(order=order):
                rig = Rig()
                rig.live_tail()
                rig.stage(True)
                rig.dev.run_all(order)
                self.assertEqual(rig.seen['n_rows'][0][:4], [5, 6, 7, 0], "n's reader is ahead of n+1's row write")
                self.assertEqual(rig.seen['n_rows'][1][:6], [18, 19, 20, 21, 0, 0])
                self.assertEqual(vals(rig.factory.tables.rows[0])[:4], [5, 6, 7, 8], "then n+1's row")

    def test_the_sources_live_until_the_event_after_the_last_copy_then_go(self):
        rig = Rig()
        rig.live_tail()
        rig.stage(True)
        keep = rig.factory.h2d_keep
        self.assertEqual(len(keep.entries), 1)
        event, hosts = keep.entries[0]
        refs = [weakref.ref(h.store) for h in hosts]
        del hosts
        self.assertEqual(len(refs), lc.AHEAD_NOSYNC['copies'])
        gc.collect()
        self.assertTrue(all(r() is not None for r in refs))
        rig.dev.run_stream(rig.dev.prefill, 3)        # n's tail, n's reader and the first copy: the event is still ahead
        rig.dev.run_stream(rig.dev.side)              # the idle streams run at once
        rig.dev.run_stream(rig.dev.default)
        self.assertFalse(event.query())
        self.assertEqual(keep.prune(), 1)
        del event
        gc.collect()
        self.assertTrue(all(r() is not None for r in refs), 'kept while copies are queued')
        rig.reader()
        rig.dev.run_all()
        self.assertEqual(rig.seen['n1'], reference_values())
        self.assertEqual(keep.prune(), 0)
        gc.collect()
        self.assertTrue(all(r() is None for r in refs), 'dropped once the event completed')
        self.assertEqual((lc.AHEAD_NOSYNC['kept'], lc.AHEAD_NOSYNC['retired']), (1, 1))

    def test_the_next_staging_prunes_and_nothing_ever_waits(self):
        rig = Rig()
        rig.stage(True)
        rig.dev.run_all()
        rig.factory.drop_ahead()
        rig.stage(True, pa.later(txn_seq=3))
        self.assertEqual(len(rig.factory.h2d_keep.entries), 1, 'the first entry retired by the second staging')
        self.assertEqual(rig.dev.syncs, [])

    def test_adopt_finishes_the_staged_chunk_with_the_off_paths_slice_three(self):
        seen = {}
        for nosync in (False, True):
            rig = Rig()
            rig.stage(nosync)
            rig.dev.run_all()
            rig.factory.complete_chunk(rig.first, consumers_complete=True)
            before = len(rig.calls)
            chunk = rig.factory.adopt(pa.later())
            self.assertIs(chunk, rig.factory._active_metadata_chunk)
            self.assertIsNone(chunk._staging, 'a later input_batch would take the ordinary path')
            seen[nosync] = [c[0] for c in rig.calls[before:]]
        self.assertEqual(seen[True], seen[False])
        self.assertIn('align-pre', seen[True])
        self.assertEqual(lc.PREPARE_AHEAD['adopted'], 2)

    def test_a_failed_build_keeps_what_it_copied_and_stages_nothing(self):
        rig = Rig()
        with mock.patch.object(lc.LaneChunk, 'build_metadata', side_effect=RuntimeError('boom')):
            self.assertEqual(rig.stage(True), dict(staged=False, why='error'))
        self.assertEqual(len(rig.factory.h2d_keep.entries), 1)
        self.assertEqual((lc.AHEAD_NOSYNC['errors'], lc.PREPARE_AHEAD['errors']), (1, 1))
        self.assertIsNone(rig.factory._staged)

    def test_the_refusals_are_the_off_paths(self):
        rig = Rig()
        self.assertEqual(rig.stage(True, tlc.admission())['why'], 'not-a-later-chunk')
        self.assertEqual(rig.stage(True, 'x')['why'], 'not-an-admission')
        rig.stage(True)
        self.assertEqual(rig.stage(True)['why'], 'already')

    def test_the_probe_runs_once_after_the_first_device_tensor_on_both_paths(self):
        for nosync in (False, True):
            with self.subTest(nosync=nosync):
                rig = Rig()
                rig.live_tail()
                marks = []
                with self.torch_scope(rig):
                    rig.factory.stage_ahead(pa.later(), nosync=nosync, probe=lambda: marks.append(list(rig.dev.done)))
                self.assertEqual(len(marks), 1)
                self.assertEqual(('prefill', 'tail-n') in marks[0], not nosync, 'off: the wait is before the probe')

    @staticmethod
    def torch_scope(rig):
        return rig.torch.cuda.stream(rig.dev.prefill)


class SlotsFast(Reset, unittest.TestCase):
    """The same staging with the inwait fast slot path selected (select_slots_fast; the box's nt rows may run it): its CPU tensor
    reaches the device by the same pinned non-blocking copy."""

    def test_the_fast_slots_are_the_slow_ones_on_the_off_path(self):
        slow = reference_values()
        lc.SLOTS_FAST['selected'] = True
        fast = reference_values()
        self.assertGreater(lc.SLOTS_FAST['calls'], 0, 'the fast path ran')
        self.assertEqual(fast, slow)

    def test_nothing_waits_and_readers_see_the_off_paths_values(self):
        want = reference_values()
        lc.SLOTS_FAST['selected'] = True
        for order in (('prefill', 'default', 'side'), ('side', 'default', 'prefill')):
            with self.subTest(order=order):
                calls = lc.SLOTS_FAST['calls']
                rig = Rig()
                rig.live_tail()
                result = rig.stage(True)
                self.assertGreater(lc.SLOTS_FAST['calls'], calls + 1, "n+1's slots took the fast path too")
                self.assertTrue(result['staged'])
                self.assertEqual(rig.dev.syncs, [])
                self.assertEqual(set(rig.dev.h2d), {('prefill', True, True)})
                self.assertFalse(rig.dev.prefill.query())
                rig.reader()
                gc.collect()
                rig.dev.run_all(order)
                self.assertEqual(rig.seen['n1'], want)


# --- stage_wire: the worker's carrier and the receipt ------------------------------------------------------------------------

class FakeStream:
    def __init__(self, busy):
        self.busy = list(busy)

    def query(self):
        return not self.busy.pop(0)


def bridge(stage, stream=None):
    dispatcher = types.SimpleNamespace(stage_ahead=stage, prefill_stream=lambda: stream)
    return types.SimpleNamespace(rank=2, dispatcher=dispatcher)


def finalize(seq=5):
    from e3_types import TransactionId
    return types.SimpleNamespace(transaction=TransactionId(1, 'req-a', 0, 4, 8, seq), op_seq=16)


class Wire(Reset, unittest.TestCase):
    def setUp(self):
        super().setUp()
        saved = sw._AHEAD_TIMED[0]
        self.addCleanup(lambda: sw._AHEAD_TIMED.__setitem__(0, saved))
        sw._AHEAD_TIMED[0] = False

    def test_the_wrapper_carries_one_record_and_pickles(self):
        rec = pa.record()
        scoped = sw.StageScoped(finalize(), (2, 3), ahead=sw.AheadNoSync(rec))
        self.assertIs(scoped.ahead.record, rec)
        back = pickle.loads(pickle.dumps(sw.StageScoped('m', (2, 3), ahead=sw.AheadNoSync(rec))))
        self.assertEqual(back.ahead, sw.AheadNoSync(rec))
        from e3_types import E3ContractError
        with self.assertRaises(E3ContractError):
            sw.AheadNoSync('not-a-record')
        self.assertEqual(tuple(sw.StageScoped.__dataclass_fields__), ('message', 'ranks', 'local_release', 'ahead'),
                         'the scoped record has no new field: the off wire bytes are unchanged')

    def test_not_installed_a_plain_record_is_staged_exactly_as_before(self):
        seen = []
        b = bridge(lambda *a, **kw: seen.append((a, kw)) or dict(staged=True, why='staged'))
        with mock.patch('native_measurement.emit_receipt_kv') as emit:
            sw.stage_rpc(b, 'submit', sw.StageScoped(finalize(), (2,), ahead=pa.record()), lambda m: 'ack')
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1], {}, 'no keyword argument reaches the dispatcher')
        emit.assert_not_called()

    def test_a_wrapped_record_is_staged_nosync_with_the_probe_and_one_receipt(self):
        seen, lines = [], []

        def stage(admission, nosync=False, probe=None):
            seen.append((admission.chunk_start, nosync))
            probe()
            return dict(staged=True, why='staged', nosync=nosync, copies=9, bytes=640)
        b = bridge(stage, FakeStream([True, True]))
        rec = pa.record()
        answer = sw.stage_rpc(b, 'submit', sw.StageScoped(finalize(), (2, 3), ahead=sw.AheadNoSync(rec)),
                              lambda m: 'ack')
        self.assertEqual(answer, 'ack')
        self.assertEqual(seen, [(8, True)])
        with mock.patch('native_measurement.emit_receipt_kv', side_effect=lambda text, logger: lines.append(text)):
            sw.timed_stage_ahead(bridge(stage, FakeStream([True, False])), 'poll',
                                 sw.StageScoped(finalize(), (2,), ahead=sw.AheadNoSync(rec)))
        self.assertEqual(len(lines), 1)
        line = lines[0]
        self.assertTrue(line.startswith(sw.AHEAD_SPLIT_LINE))
        kv = dict(part.split('=', 1) for part in line[len(sw.AHEAD_SPLIT_LINE):].split())
        self.assertEqual({k: kv[k] for k in ('rank', 'txn_seq', 'chunk', 'next', 'rpc', 'op_seq', 'nosync', 'why', 'busy_in',
                                             'busy_out', 'copies', 'bytes')},
                         dict(rank='2', txn_seq='5', chunk='4-8', next='8-11', rpc='poll', op_seq='16', nosync='1',
                              why='staged', busy_in='1', busy_out='0', copies='9', bytes='640'))
        for k in ('wait_ns', 'wait_cpu_ns', 'work_ns', 'work_cpu_ns'):
            self.assertGreaterEqual(int(kv[k]), 0, k)

    def test_installed_a_plain_record_is_timed_on_the_off_path(self):
        sw._AHEAD_TIMED[0] = True
        seen, lines = [], []

        def stage(admission, nosync=False, probe=None):
            seen.append(nosync)
            probe()
            return dict(staged=True, why='staged')
        with mock.patch('native_measurement.emit_receipt_kv', side_effect=lambda text, logger: lines.append(text)):
            sw.stage_rpc(bridge(stage), 'submit', sw.StageScoped(finalize(), (2,), ahead=pa.record()), lambda m: 'ack')
        self.assertEqual(seen, [False])
        self.assertIn(' nosync=0 ', lines[0])
        self.assertIn(' busy_in=-1 busy_out=-1 ', lines[0], 'no stream: unknown')

    def test_no_build_no_line_and_a_failing_stage_never_reaches_the_command(self):
        lines = []
        with mock.patch('native_measurement.emit_receipt_kv', side_effect=lambda text, logger: lines.append(text)):
            sw.stage_rpc(bridge(lambda a, nosync=False, probe=None: dict(staged=True, why='already')), 'poll',
                         sw.StageScoped(finalize(), (2,), ahead=sw.AheadNoSync(pa.record())), lambda m: 'ack')

            def boom(a, nosync=False, probe=None):
                raise RuntimeError('x')
            self.assertEqual(sw.stage_rpc(bridge(boom), 'poll', sw.StageScoped(finalize(), (2,), ahead=sw.AheadNoSync(pa.record())),
                                          lambda m: 'ack'), 'ack')
        self.assertEqual(lines, [])
        self.assertEqual(lc.PREPARE_AHEAD['errors'], 1)

    def test_the_setting_reads_strictly(self):
        self.assertFalse(sw.ahead_nosync_installed({}))
        self.assertFalse(sw.ahead_nosync_installed({'FLASH_S5_AHEAD_NOSYNC': '0'}))
        self.assertTrue(sw.ahead_nosync_installed({'FLASH_S5_AHEAD_NOSYNC': '1'}))
        from e3_types import E3ContractError
        with self.assertRaises(E3ContractError):
            sw.ahead_nosync_installed({'FLASH_S5_AHEAD_NOSYNC': 'yes'})


class ThroughTheWriter(Reset, unittest.TestCase):
    """parity-na2: q18a-parity-o3 n1 had no ahead-split line at all. The box installs the process receipt writer, whose allow-list
    refused the prefix (a fault, nothing written), and the tests above mock emit_receipt_kv away. Here the real writer, the real
    receipt parser and the box grade's own reader over the written bytes."""

    def setUp(self):
        super().setUp()
        import native_measurement as nm
        self.nm, self.out = nm, []
        self.writer = nm.ReceiptWriter(fd=9, write=lambda fd, data: (self.out.append(data), len(data))[1], getpid=lambda: 77)
        with mock.patch.object(nm, 'receipt_path', lambda fd, limit=None: 'atomic-fifo'):
            self.writer._start()
        patcher = mock.patch.object(nm, '_RECEIPT_WRITER', self.writer)
        patcher.start()
        self.addCleanup(patcher.stop)

    def emit(self):
        def stage(admission, nosync=False, probe=None):
            probe()
            return dict(staged=True, why='staged', nosync=nosync, copies=9, bytes=640)
        sw.timed_stage_ahead(bridge(stage, FakeStream([True, True])), 'state_finalize',
                             sw.StageScoped(finalize(), (2,), ahead=sw.AheadNoSync(pa.record())))

    def test_the_line_is_written_whole_and_the_parser_reads_it(self):
        import receipt_lines as rl
        before = self.writer.seq
        self.emit()
        self.assertEqual((self.writer.faults, self.writer.seq, lc.AHEAD_NOSYNC['unwritten']), (0, before + 1, 0))
        line = self.out[-1].decode()
        self.assertTrue(line.startswith(sw.AHEAD_SPLIT_LINE + 'rank=2 txn_seq=5 chunk=4-8 next=8-11 rpc=state_finalize '), line)
        record = rl.parse_line(line)
        self.assertEqual((record['pid'], record['seq'], record['sha']), (77, before, rl.sha16([record['text']])))
        self.assertEqual(rl.KV_PREFIXES, self.nm.RECEIPT_KV_PREFIXES, 'the parser knows every writer prefix')

    def test_the_box_grade_counts_the_written_line_in_its_arm(self):
        import e3_compose5_box as box
        import tempfile
        from pathlib import Path
        self.emit()
        written = self.out[-1].decode()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'engine-stream.log'
            log.write_text('2026-11-07T13:40:00.500000000Z ' + written)       # docker's stamp, as the record's stream
            at = box.log_wall('2026-11-07T13:40:00.500000000Z x')
            arms = [dict(label='na.g2-live-0', wall_start_s=at - 10, wall_end_s=at + 10), dict(label='nt.g2-live-0', wall_start_s=at + 20, wall_end_s=at + 40)]
            read = box.ahead_split([str(log)], arms)
        self.assertEqual((read['arms']['na.g2-live-0']['lines'], read['arms']['na.g2-live-0']['nosync']), (1, 1))
        self.assertEqual(read['arms']['nt.g2-live-0']['nosync'], 0)

    def test_a_refused_line_is_counted_unwritten(self):
        with mock.patch.object(self.nm, 'RECEIPT_KV_PREFIXES', ('flash-s5 memory ',)):
            self.emit()
        self.assertEqual((self.writer.faults, lc.AHEAD_NOSYNC['unwritten']), (1, 1))
        self.assertFalse([d for d in self.out if b'ahead-split' in d])


class Dispatcher(Reset, unittest.TestCase):
    """lane_dispatch.stage_ahead passes nosync/probe only when set, under the same refusals and the lane stream."""

    def test_keywords_only_when_set(self):
        import lane_dispatch
        seen = []
        f = types.SimpleNamespace(stage_ahead=lambda *a, **kw: seen.append(kw) or dict(staged=True, why='staged'),
                                  _refuse_ahead=lambda why: dict(staged=False, why=why))
        d = types.SimpleNamespace(factory=f, quarantined=None, transaction=None, _slices=None, accepted={},
                                  _scope=contextlib.nullcontext)
        lane_dispatch.LaneDispatcher.stage_ahead(d, 'adm')
        d._stage_ahead_with = lambda adm, nosync, probe: lane_dispatch.LaneDispatcher._stage_ahead_with(d, adm, nosync, probe)
        lane_dispatch.LaneDispatcher.stage_ahead(d, 'adm', nosync=True)
        probe = object()
        lane_dispatch.LaneDispatcher.stage_ahead(d, 'adm', probe=probe)
        self.assertEqual(seen, [{}, dict(nosync=True, probe=None), dict(nosync=False, probe=probe)])
        d.transaction, d.accepted = 'txn', {}
        self.assertEqual(lane_dispatch.LaneDispatcher.stage_ahead(d, 'adm', nosync=True)['why'], 'live-chunk-not-finalized')


# --- the engine: carrier and selection ---------------------------------------------------------------------------------------

class Carry(unittest.TestCase):
    def sender(self, nosync, counters=True):
        import stage_progress as sp
        rec = pa.record()
        frame_k = types.SimpleNamespace(dead=False, opened={0, 1})
        nxt = types.SimpleNamespace(dead=False, opened=set(),
                                    stage={0: dict(open_request=types.SimpleNamespace(admission=rec)),
                                           1: dict(open_request=types.SimpleNamespace(admission=rec))})
        driver = types.SimpleNamespace(prepare_ahead=True, _scoped_frame=frame_k, _scoped_stage=1, _frames=[frame_k, nxt])
        driver.__dict__['phase'] = 'finalize'
        if nosync is not None:
            driver.__dict__['ahead_nosync'] = nosync
        if counters:
            driver.__dict__['ahead_nosync_counters'] = dict(sent=0)
        return sp._ScopedSender(driver, None, (2, 3)), rec, driver

    def fin(self):
        import wire_v1 as w
        return types.SimpleNamespace(kind=w.SegmentKind.STATE_FINALIZE)

    def test_selected_the_record_is_wrapped_and_counted(self):
        s, rec, driver = self.sender(True)
        out = s._ahead('flash_s5_submit', self.fin())
        self.assertEqual(out, sw.AheadNoSync(rec))
        self.assertIs(out.record, rec)
        self.assertIsInstance(s._ahead('flash_s5_poll', 'p'), sw.AheadNoSync, "stage 0's poll carrier too")
        self.assertEqual(driver.ahead_nosync_counters, dict(sent=2))

    def test_off_or_not_installed_the_record_is_the_same_object(self):
        for nosync in (False, None):
            with self.subTest(nosync=nosync):
                s, rec, _driver = self.sender(nosync, counters=nosync is not None)
                self.assertIs(s._ahead('flash_s5_submit', self.fin()), rec)

    def test_no_successor_no_wrapper(self):
        s, _rec, _driver = self.sender(True)
        s.driver._frames = s.driver._frames[:1]
        self.assertIsNone(s._ahead('flash_s5_submit', self.fin()))


import stage_progress as _sp  # noqa: E402
import test_e3_stage_overlap as _c1  # noqa: E402
import test_inwait_engine as _ie  # noqa: E402
from stage_net import ed as _ed  # noqa: E402


class Selection(_ie.FlagOn, _ie.Early, _c1.OverlapBase):
    def idle(self, installed=True, ahead=True):
        network = self.network()
        driver = network.driver
        driver.controller = types.SimpleNamespace(active=False, overlap_counters={}, overlap_refusals={},
                                                  exact_counters=dict(exact_admits=0, exact_terminal=0),
                                                  terminal_counters=dict(terminal_admits=0, terminal_mismatch=0))
        driver.engine = types.SimpleNamespace(is_scheduler_paused=lambda: True)
        driver.ahead_nosync_installed = installed
        driver.prepare_ahead = ahead
        return driver

    base = dict(action=_sp.STAGE_CONTROL, local=True, overlap=True, defer_bind=True, inwait=True, early_successor=True)

    def test_off_until_selected_echoed_and_in_status(self):
        driver = self.idle()
        self.assertIs(driver.ahead_nosync, False)
        reply = driver._stage_control(dict(self.base, ahead_nosync=True))
        self.assertEqual((reply['ahead_nosync'], reply['previous_ahead_nosync']), (True, False))
        self.assertIs(driver.ahead_nosync, True)
        self.assertEqual(driver.stage_status()['overlap']['ahead_nosync'], dict(installed=True, selected=True, counters=dict(sent=0)))
        self.assertIs(driver._stage_control(dict(self.base))['ahead_nosync'], False, 'a command without the field turns it off')

    def test_refused_when_invalid_not_installed_or_without_prepare_ahead(self):
        for kwargs, bad in ((dict(), dict(self.base, ahead_nosync=1)),
                            (dict(installed=False), dict(self.base, ahead_nosync=False)),
                            (dict(ahead=False), dict(self.base, ahead_nosync=True))):
            with self.subTest(bad=bad, kwargs=kwargs):
                driver = self.idle(**kwargs)
                with self.assertRaises(_ed.DriverError):
                    driver._stage_control(bad)
                self.assertIs(driver.ahead_nosync, False, 'nothing applied on a refusal')

    def test_not_installed_the_reply_and_status_are_unchanged(self):
        driver = self.idle(installed=False)
        reply = driver._stage_control(dict(self.base))
        self.assertNotIn('ahead_nosync', reply)
        self.assertNotIn('ahead_nosync', driver.stage_status()['overlap'])

    def test_the_driver_reads_the_launch_setting(self):
        import stage_overlap as so
        self.assertFalse(so.ahead_nosync_enabled({}))
        self.assertTrue(so.ahead_nosync_enabled({'FLASH_S5_AHEAD_NOSYNC': '1'}))
        with self.assertRaises(_ed.DriverError):
            so.ahead_nosync_enabled({'FLASH_S5_AHEAD_NOSYNC': '2'})


class Control(unittest.TestCase):
    ENV = dict(FLASH_S5_STAGE_LOCAL='1', FLASH_S5_AUTO_ARM='1', FLASH_S5_STAGE_OVERLAP='1', FLASH_S5_STAGE_OVERLAP_DEFER_BIND='1', FLASH_S5_INWAIT='1')

    def call(self, env, reply_extra, **kwargs):
        import native_overlap_control as noc
        sent = []

        async def _call(command):
            sent.append(command)
            return dict(local=True, overlap=True, defer_bind=True, inwait=True, **reply_extra)
        client = types.SimpleNamespace(engine_epoch=1, _call=_call)
        with mock.patch.dict(os.environ, env, clear=False):
            for name in ('FLASH_S5_AHEAD_NOSYNC',):
                if name not in env:
                    os.environ.pop(name, None)
            asyncio.run(noc.NativeControlClient.native_stage_overlap(client, True, True, True, inwait=True, **kwargs))
        return sent

    def test_sent_only_installed_and_the_echo_checked(self):
        import native_overlap_control as noc
        with self.assertRaises(noc.NativeControlError):
            self.call(self.ENV, dict(ahead_nosync=True), ahead_nosync=True)
        on = dict(self.ENV, FLASH_S5_AHEAD_NOSYNC='1')
        self.assertIs(self.call(on, dict(ahead_nosync=True), ahead_nosync=True)[0]['ahead_nosync'], True)
        with self.assertRaises(noc.NativeControlError):
            self.call(on, dict(ahead_nosync=False), ahead_nosync=True)
        with self.assertRaises(noc.NativeControlError):
            self.call(on, dict(ahead_nosync=1), ahead_nosync=1)
        self.assertNotIn('ahead_nosync', self.call(self.ENV, {})[0], 'not sent: the command is as before')


if __name__ == '__main__':
    unittest.main()
