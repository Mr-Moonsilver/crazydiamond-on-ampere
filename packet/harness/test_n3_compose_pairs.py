#!/usr/bin/env python3
"""Composition branch (lead 12 Nov 2026): lever pairs that share a code path, run together on the real engine code.

Y1 (FLASH_S5_YIELD_NO_APPEND_OWNER) x Y2 (FLASH_S5_YIELD_WINDOW), on test_yield_window's rig (the real auto-arm
policy, engine driver and scheduler hooks over the CPU worker stack; O3 not installed): Y1 decides at the yield offer,
before Y2's window exists, from the owner's rows left past the transaction frontier.
  * owner rows left > max_new: Y1 lets the yield through, and the window runs exactly as with Y1 off (same window
    counters, the same append-yield event kinds, the same receipts), Y1 counting no junction;
  * owner rows left <= max_new: Y1 refuses the yield, so no window opens (Y2's counters stay zero), the junction is
    counted once, and the owner keeps its lane.
The rig's bulk has 7 rows left at its first junction (prompt 12, frontier 4, native end 11); max_new 6 and 8 put it
on either side.

Y2 x lever A (FLASH_S5_APPEND_NO_BLOCK_STOP), on the same window rig with stock's own _mamba_block_aligned_split (the
reference text, test_coverage_residue.SPLIT) on its scheduler: the window's APPEND (11 tokens, 7 cached) starts inside
Mamba block 1, so stock stops its native piece at the block boundary 8; lever A runs it on to the native end 10 in one
piece. Geometry: block 4 (the bulk's pieces 0-4-8-11 and the waiter's 0-4-5 start on block boundaries, so stock's split
leaves them as they are), hash unit 2, long_prefill_token_threshold 2 (a block wider than a prefill chunk, as at
production, block 4048 over budget 1024: stock advances sub-block, no aligned-end floor). With A on and off: the same
window counters, append-yield event kinds and receipts; with A on the APPEND runs (7, 10) instead of (7, 8), (8, 10),
one fused piece and one stale block unpublished, the call counters showing lever A's split decided it.
D (FLASH_S5_DECODE_SAMPLE_FAST, with FLASH_S5_DECODE_ANATOMY) x Y2 and x lever A: D shares the engine step seam
(executor_hooks.before_step_with_batch_queue runs decode_step.EngineSide.step before auto-arm's tick) and reads the
policy (manual_owned, active) and the driver's executor (_flash_s5_progress_recording); it writes neither. Both flows
above run again with an EngineSide stepped on the real policy before every policy tick: the same window counters,
event kinds, receipts and (for A) pieces and piece counters as without it; D's gate on (the native arm owns the lane),
sample_tokens tagged fast, and the anatomy window never eligible (the policy is active and native progress records
throughout), so no anatomy line. Per step the policy's fields, the driver's yield phase and transaction and the queues are compared
before and after D's step. Mutants: an inverted gate and a D that writes the policy (manual_owned) are caught; the
Y2 end state alone does not show that write in this rig, the per-step comparison does.
CPU only; no box, GPU or torch device.
"""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE.parent / 'patches' / 's5_engine'):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))

import types  # noqa: E402
from types import SimpleNamespace as NS  # noqa: E402
from unittest import mock  # noqa: E402

import decode_step as ds  # noqa: E402
import native_auto_arm as auto  # noqa: E402
import scheduler_hooks as sh  # noqa: E402
import test_coverage_residue as tcr  # noqa: E402
import test_yield_window as yw  # noqa: E402


class Y1WithY2(unittest.TestCase):
    def flow(self, y1, max_new):
        log = yw.Lines(self)
        rig = yw.window_rig(max_new=max_new)
        rig.c.yield_no_append_owner = y1
        yw.served_first(rig)
        yw.to_window_append(rig)
        # receipts without their clock stamps (wall / monotonic values) and measured gaps
        receipts = [{k: v for k, v in r.items() if k != 'gap_ms' and not (type(v) in (int, float) and v > 10 ** 6)}
                    for r in log.receipts()]
        return rig, yw.counts(rig), yw.kinds(rig), receipts

    def test_owner_past_max_new_window_unchanged(self):
        _, off_counts, off_kinds, off_receipts = self.flow(False, 6)
        rig, on_counts, on_kinds, on_receipts = self.flow(True, 6)
        self.assertEqual((off_counts['appends'], off_counts['reentries_avoided']), (1, 1), off_counts)   # in the window
        self.assertEqual(on_counts, off_counts)
        self.assertEqual(on_kinds, off_kinds)
        self.assertEqual(on_receipts, off_receipts)
        self.assertEqual(rig.c.yield_owner_counts['junctions'], 0, rig.c.yield_owner_counts)
        self.assertEqual(rig.d.append_yield_phase(), 'append')

    def test_owner_within_max_new_no_window(self):
        yw.Lines(self)
        rig = yw.window_rig(max_new=8)
        rig.c.yield_no_append_owner = True
        yw.ay.run_to_yield(rig)
        with self.assertRaises(AssertionError):
            yw.ay.to_append(rig)                  # the yield never starts: Y1 refuses at every offer
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertEqual(rig.c.yield_owner_counts['junctions'], 1, rig.c.yield_owner_counts)
        self.assertEqual(rig.c.yield_owner_counts['errors'], 0)
        self.assertEqual({k: yw.counts(rig)[k] for k in ('windows', 'appends', 'reentries_avoided')},
                         dict(windows=0, appends=0, reentries_avoided=0))

    def test_control_without_y1_the_same_rig_yields(self):
        yw.Lines(self)
        rig = yw.window_rig(max_new=8)
        rig.c.yield_no_append_owner = False
        yw.served_first(rig)
        yw.to_window_append(rig)
        self.assertEqual(yw.counts(rig)['appends'], 1)


class MambaGroup:
    """An align-mode Mamba group as commit's unpublish sees it (the cached-block counts cache_blocks reads)."""
    mamba_cache_mode = 'align'

    def __init__(self):
        self.num_cached_block = {}


def mamba_split(rig, block=4, unit=2, threshold=2):
    """Stock's split on the rig's scheduler, with the fields it reads; the requests' shared_prefix_boundary is 0."""
    for name, value in tcr.split_host(cache_config=NS(block_size=block), hash_block_size=unit,
                                      mamba_prefill_checkpoint_alignment=block,
                                      max_num_scheduled_tokens=rig.s.max_num_scheduled_tokens,
                                      scheduler_config=NS(long_prefill_token_threshold=threshold)).items():
        setattr(rig.s, name, value)
    rig.s._mamba_block_aligned_split = types.MethodType(tcr.SPLIT, rig.s)
    group = MambaGroup()
    rig.s.kv_cache_manager.coordinator.single_type_managers = (group,)
    return group


class Y2WithA(unittest.TestCase):
    """The window's APPEND through stock's block stop, lever A off and on (see the module docstring)."""

    def setUp(self):
        patch = mock.patch.object(yw.adm.Request, 'shared_prefix_boundary', 0, create=True)
        patch.start()
        self.addCleanup(patch.stop)
        self.calls = {'aligned_split': 0, 'block_fused': []}
        aligned, fused = sh.PrefillAdmissionController._aligned_split, sh.PrefillAdmissionController._append_block_fused

        def aligned_split(controller, *a, **k):
            self.calls['aligned_split'] += 1
            return aligned(controller, *a, **k)

        def block_fused(controller, scheduler, splitter, request, limit, local, size):
            value = fused(controller, scheduler, splitter, request, limit, local, size)
            self.calls['block_fused'].append((request.request_id, request.num_computed_tokens + local, size, value))
            return value
        for name, fn in (('_aligned_split', aligned_split), ('_append_block_fused', block_fused)):
            patch = mock.patch.object(sh.PrefillAdmissionController, name, fn)
            patch.start()
            self.addCleanup(patch.stop)

    def flow(self, a):
        self.calls['aligned_split'], self.calls['block_fused'] = 0, []
        log = yw.Lines(self)
        rig = yw.window_rig()
        group = mamba_split(rig)
        rig.c.append_no_block_stop = a
        append = yw.ay.run_to_yield(rig, append_cached=7)
        group.num_cached_block[append.request_id] = 7 // 4      # the hit's cached blocks (block 0)
        rig.add_held(yw.OTHER_UID, 6)
        yw.ay.to_append(rig)
        pieces = [(t.chunk_start, t.chunk_end) for t in yw.ay.run_append(rig)]
        yw.ay.to_full(rig)
        rig.stock_handback(append)
        yw.to_window_append(rig)
        receipts = [{k: v for k, v in r.items() if k != 'gap_ms' and not (type(v) in (int, float) and v > 10 ** 6)}
                    for r in log.receipts()]
        counts = {k: v for k, v in yw.counts(rig).items() if k != 'gap_ms'}
        return rig, pieces, counts, yw.kinds(rig), receipts, dict(self.calls, block_fused=list(self.calls['block_fused'])), group

    def test_window_unchanged_and_the_append_fused(self):
        _, off_pieces, off_counts, off_kinds, off_receipts, off_calls, _ = self.flow(False)
        rig, on_pieces, on_counts, on_kinds, on_receipts, on_calls, group = self.flow(True)
        self.assertEqual((off_counts['appends'], off_counts['reentries_avoided']), (1, 1), off_counts)   # in the window
        self.assertEqual(on_counts, off_counts)
        self.assertEqual(on_kinds, off_kinds)
        self.assertEqual(on_receipts, off_receipts)
        self.assertEqual(rig.d.append_yield_phase(), 'append')
        self.assertEqual(off_pieces, [(7, 8), (8, 10)])                # stock's block stop at 8
        self.assertEqual(on_pieces, [(7, 10)])                         # lever A: through it to the native end
        # witness: stock's split reached with both settings; lever A's split ran only with A on and fused at 7
        self.assertGreater(off_calls['aligned_split'], 0)
        self.assertGreater(on_calls['aligned_split'], 0)
        self.assertEqual(off_calls['block_fused'], [])
        self.assertEqual({c for c in on_calls['block_fused'] if c[3] != c[2]}, {(yw.APPEND_UID, 7, 1, 3)})  # plan + admit
        pieces = rig.c.append_pieces
        self.assertEqual((pieces['fused_pieces'], pieces['stops_skipped'], pieces['appends'], pieces['hist']),
                         (1, 1, 1, {'1': 1}), pieces)
        self.assertEqual((pieces['stale_states_unpublished'], pieces['errors'], pieces['fallbacks']), (1, 0, {}), pieces)
        self.assertEqual(group.num_cached_block[yw.APPEND_UID], 2)    # block 1 (the state at 7) never hashed

    def test_control_without_the_splitter_lever_a_is_not_reached(self):
        """The gap this class closes: the plain window rig has no Mamba split, so A's code never runs."""
        yw.Lines(self)
        rig = yw.window_rig()
        rig.c.append_no_block_stop = True
        yw.served_first(rig)
        yw.to_window_append(rig)
        self.assertGreater(self.calls['aligned_split'], 0)
        self.assertEqual(self.calls['block_fused'], [])
        self.assertEqual(rig.c.append_pieces['fused_pieces'], 0)


class EngineView:
    """The engine as before_step_with_batch_queue hands it to EngineSide.step: the policy's own engine and driver."""

    def __init__(self, policy):
        self.policy = policy

    @property
    def _flash_s5_driver(self):
        return self.policy.driver

    @property
    def model_executor(self):
        return self.policy.driver.executor

    @property
    def scheduler(self):
        return self.policy.engine.scheduler

    @property
    def batch_queue(self):
        return self.policy.engine.batch_queue


def _plain(value):
    return value if isinstance(value, (bool, int, float, str, tuple, type(None))) else ('obj', id(value))


def engine_state(policy):
    """What D's step could reach: the policy's fields, the driver's yield phase and transaction, the queues."""
    scheduler = policy.engine.scheduler
    return ({k: _plain(v) for k, v in vars(policy).items()}, policy.driver.append_yield_phase(),
            id(policy.driver.tx), policy.driver.error,
            [r.request_id for r in scheduler.running], [r.request_id for r in scheduler.waiting.items],
            [r.request_id for r in scheduler.skipped_waiting.items], len(policy.engine.batch_queue))


class DStepped:
    """Mixin: every policy tick of a flow is preceded by EngineSide.step (when ``self.side`` is set); per step the
    gate, the window's eligibility inputs and the stock tags D would put on sample_tokens / execute_model."""
    side = None
    module = ds

    def setUp(self):
        super().setUp()
        self.trace, self.ticks, self.writes = [], 0, []
        tick = auto.AutoArmPolicy.tick
        test = self

        def stepped(policy, *a, **k):
            test.ticks += 1
            if test.side is not None:
                view = EngineView(policy)
                before = engine_state(policy)
                test.side.step(view)
                if engine_state(policy) != before:
                    test.writes.append(len(test.trace))
                test.trace.append(dict(
                    native=test.side.native_arm, active=bool(policy.active),
                    progress=getattr(view.model_executor, '_flash_s5_progress_recording', None) is not None,
                    sample=test.side.wire('sample_tokens', len(test.trace)),
                    execute=test.side.wire('execute_model', len(test.trace))))
            return tick(policy, *a, **k)
        patch = mock.patch.object(auto.AutoArmPolicy, 'tick', stepped)
        patch.start()
        self.addCleanup(patch.stop)

    def with_d(self, flow, *args):
        self.trace, self.lines, self.writes = [], [], []
        logger = NS(info=lambda fmt, *a: self.lines.append(fmt % a), warning=lambda *a, **k: None)
        self.side = self.module.EngineSide({ds.SAMPLE_FAST_ENV: '1', ds.ANATOMY_ENV: '1'}, logger=logger)
        try:
            return flow(*args)
        finally:
            self.side = None

    def assert_d_inert(self):
        self.assertGreater(len(self.trace), 0)
        self.assertEqual(self.writes, [])                                    # D's step wrote no shared state
        self.assertEqual(self.side_counters['errors'], 0)
        self.assertTrue(all(t['native'] for t in self.trace), self.trace[:3])    # the native arm owns the lane
        self.assertTrue(all(t['sample'] == [1, ds.TAG_NONE, i] for i, t in enumerate(self.trace)))
        self.assertTrue(all(t['execute'] is None for t in self.trace))           # no window: nothing else tagged
        self.assertTrue(all(t['active'] or t['progress'] for t in self.trace))   # never a clean decode window
        self.assertEqual([x for x in self.lines if x.startswith(ds.LINE_ANATOMY)], [])
        self.assertEqual(len([x for x in self.lines if x.startswith(ds.LINE_GATE)]), 1)   # the one gate-on line


def _strip(flow_result):
    """A flow's comparable part: everything but the rig and the Mamba group objects."""
    return [x for x in flow_result if not isinstance(x, (yw.Rig, MambaGroup))]


class DWithY2(DStepped, unittest.TestCase):
    def flow(self):
        self.ticks = 0
        log = yw.Lines(self)
        rig = yw.window_rig(max_new=6)
        yw.served_first(rig)
        yw.to_window_append(rig)
        receipts = [{k: v for k, v in r.items() if k != 'gap_ms' and not (type(v) in (int, float) and v > 10 ** 6)}
                    for r in log.receipts()]
        counts = {k: v for k, v in yw.counts(rig).items() if k != 'gap_ms'}
        return rig, counts, yw.kinds(rig), receipts, rig.d.append_yield_phase(), self.ticks

    def test_window_unchanged_with_d(self):
        off = self.flow()
        self.assertEqual(self.trace, [])
        on = self.with_d(self.flow_capture)
        self.assertEqual((off[1]['appends'], off[1]['reentries_avoided']), (1, 1), off[1])
        self.assertEqual(_strip(on), _strip(off))
        self.assert_d_inert()

    def flow_capture(self):
        try:
            return self.flow()
        finally:
            self.side_counters = dict(self.side.counters)

    def mutant(self, old, new):
        source = (HERE.parent / 'patches/s5_engine/decode_step.py').read_text()
        self.assertEqual(source.count(old), 1, old)
        module = types.ModuleType('decode_step_compose_mutant')
        exec(compile(source.replace(old, new), 'decode_step_compose_mutant.py', 'exec'), module.__dict__)
        return module

    def test_mutant_inverted_gate_is_caught(self):
        """D reading the policy wrong (fast only while stock owns the lane) shows in the composed rig."""
        self.module = self.mutant("policy.manual_owned is False\n", "policy.manual_owned is True\n")
        try:
            self.with_d(self.flow_capture)
            with self.assertRaises(AssertionError):
                self.assert_d_inert()
        finally:
            self.module = ds

    def test_mutant_d_writing_the_policy_is_caught(self):
        """D writing the policy (manual_owned set on its step): the composed flow sees the write."""
        old = "            native = policy is not None and policy.manual_owned is False\n"
        self.module = self.mutant(old, old + "            policy.manual_owned = True\n")
        try:
            self.with_d(self.flow_capture)
        finally:
            self.module = ds
        self.assertEqual(self.writes, [0])               # the first step's write; later steps find it already set
        with self.assertRaises(AssertionError):
            self.assert_d_inert()


class DWithA(DStepped, Y2WithA):
    def test_window_and_fused_append_unchanged_with_d(self):
        for a in (False, True):
            self.trace, self.ticks = [], 0
            off = self.flow(a) + (self.ticks,)
            self.assertEqual(self.trace, [])
            self.ticks = 0
            on = self.with_d(self.capture, a) + (self.ticks,)
            self.assertEqual(_strip(on), _strip(off), a)
            self.assert_d_inert()
            pieces = on[1]
            self.assertEqual(pieces, [(7, 10)] if a else [(7, 8), (8, 10)])   # A's effect is still visible with D

    def capture(self, a):
        try:
            return self.flow(a)
        finally:
            self.side_counters = dict(self.side.counters)

    # the inherited lever A tests run once in Y2WithA; here only the D pair
    test_window_unchanged_and_the_append_fused = None
    test_control_without_the_splitter_lever_a_is_not_reached = None


if __name__ == '__main__':
    unittest.main()
