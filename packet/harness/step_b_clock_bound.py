#!/usr/bin/env python3
"""CLOCK_BOUND_FROZEN for Step B: every real-clock window-slot test in the strict list and the bundle self-run, left out by id.

Each listed test executes (or asserts) a frozen queue draft's owner-window slot at real wall clock against that queue's budget and
hard end. Step B's own draft has no such slot any more (no absolute time in its gate, draft or runner), so its test is gone. Box timing is governed by the lead at launch (operator, 2 Nov 2026 about 03:05 CET), so these tests are not
gates of this package. The list is deterministic, not cut by the time of the run. The test files stay byte-identical; only these ids
are left out, each with the declared reason.

Uses: `python3 step_b_clock_bound.py MODULE [MODULE ...]` runs those test modules minus the listed ids (selftest rows, the gate
chain's focused list); `filtered(suite)` for the bundle self-run probe. Both refuse unless every listed id of the loaded modules
was found and removed exactly once.
"""
import sys
import unittest

REASON = 'box timing is governed by the lead at launch (operator, 2 Nov)'
CLOCK_BOUND_FROZEN = {name: REASON for name in (
    'test_native34_queue.FrozenProtocol.test_every_draft_window_slot_executes_at_real_wall_clock',
    'test_native35_queue.Window.test_every_native35_draft_window_slot_executes_at_real_wall_clock',
    'test_native35_performance_queue.Window.test_p_draft_window_slot_executes_at_real_wall_clock',
    'test_native35_l2.Queue.test_l2_draft_window_slot_executes_at_real_wall_clock',
    'test_native39_trunk_queue.Window.test_trunk_draft_window_slot_executes_at_real_wall_clock',
    'test_native39_lifecycle_queue.Window.test_lifecycle_draft_window_slot_executes_at_real_wall_clock',
    'test_native40_box_queue.Queue.test_exact_derivation_shell_and_executable_window',
    'test_native41_box_queue.Window.test_draft_window_slot_executes_at_real_wall_clock',
    'test_native41_progress_queue.Window.test_draft_window_slot_executes_at_real_wall_clock',
    'test_native43_boundary_queue.QueueTests.test_window_uses_same_budget_real_wall_time_and_edges',
    'test_native43_lifecycle_queue.Window.test_lifecycle_draft_window_slot_executes_at_real_wall_clock',
    'test_native43_lf_queue.Window.test_l_f_draft_window_slot_executes_at_real_wall_clock',
    'test_alpha_perf_queue.Window.test_draft_window_slot_executes_at_real_wall_clock',
    'test_step_a_queue.Window.test_draft_window_slot_executes_at_real_wall_clock',
)}
MODULES = sorted({name.split('.', 1)[0] for name in CLOCK_BOUND_FROZEN})
# The bundle self-run probe's suites hold these (native39 lifecycle, native43 boundary, lifecycle and l-f are strict-only).
IN_PROBE = sorted(name for name in CLOCK_BOUND_FROZEN if name.split('.', 1)[0] not in (
    'test_native39_lifecycle_queue', 'test_native43_boundary_queue', 'test_native43_lifecycle_queue', 'test_native43_lf_queue'))


def _flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


def filtered(suite):
    """The suite without the listed ids, and the ids removed; refuses a listed id found twice."""
    tests = list(_flatten(suite))
    removed = sorted(t.id() for t in tests if t.id() in CLOCK_BOUND_FROZEN)
    assert len(removed) == len(set(removed)), ('clock-bound id loaded twice', removed)
    return unittest.TestSuite(t for t in tests if t.id() not in CLOCK_BOUND_FROZEN), removed


def expected_removed(modules):
    return sorted(name for name in CLOCK_BOUND_FROZEN if name.split('.', 1)[0] in modules)


def main(modules):
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite(loader.loadTestsFromName(name) for name in modules)
    kept, removed = filtered(suite)
    assert removed == expected_removed(modules), ('declared clock-bound ids not found exactly once', removed)
    for name in removed:
        print('clock-bound frozen, excluded (declared): %s: %s' % (name, CLOCK_BOUND_FROZEN[name]))
    result = unittest.TextTestRunner(verbosity=1).run(kept)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    assert len(sys.argv) > 1, 'usage: step_b_clock_bound.py MODULE [MODULE ...]'
    sys.exit(main([name[:-3] if name.endswith('.py') else name for name in sys.argv[1:]]))
