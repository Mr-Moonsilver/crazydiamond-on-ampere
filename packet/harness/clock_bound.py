#!/usr/bin/env python3
"""Real-clock window-slot tests, excluded from C0's strict, focused and bundle self-run sets by one declared list.

Reason (operator, 2 Nov 2026, about 03:05 CET): box timing is governed by the lead at launch. Each listed test runs a queue draft's
PY_SLOT at the real clock against that queue's own gate, so it starts failing at a fixed instant of the night whatever the code
(native39 trunk after 01:14:59Z, native35 p after 01:44:59Z, ... native34 after 05:17:59Z). The list is every real-clock
window-slot test of the frozen and inherited queues, found by the tests' own text (PY_SLOT / SLOT_RECEIPTS with real_wall_clock,
and window_ok(time.time()) asserted directly), not cut at a time. Those test files stay byte-identical; nothing here edits a test.
The native43 boundary id (window_ok(time.time()) asserted directly, last true 03:57:20Z) was added after chain j's engine
self-run attempt 2 failed on it (lead's go, 2 Nov); a wider scan of every harness test for real-clock window asserts found no other.
Nothing listed tests C0's own gate: C0 has no absolute-time window (operator, 2 Nov: "we don't put in code what you govern"),
and its own slot test was reworked to that contract in test_alpha_c0_queue.py (C0's file), so its id left this list (chain l).

    python3 clock_bound.py SUITE.py          run one suite file as `python3 SUITE.py` would, minus its listed ids
    python3 clock_bound.py -m MOD [MOD ...]  run modules as `python3 -m unittest MOD ...` would, minus the listed ids
    python3 clock_bound.py -v -m MOD [MOD ...]  the same at verbosity 2, as `python3 -m unittest -v MOD ...` would (the engine
                                             self-run's dense suites: its runner counts '... skipped' lines)

Each run prints the ids it excluded (`clock-bound excluded: ...`), so the log shows exactly what did not run.
"""
import importlib
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REASON = 'box timing is governed by the lead at launch (operator, 2 Nov)'
CLOCK_BOUND_FROZEN = (
    'test_alpha_perf_queue.Window.test_draft_window_slot_executes_at_real_wall_clock',
    'test_native34_queue.FrozenProtocol.test_every_draft_window_slot_executes_at_real_wall_clock',
    'test_native35_l2.Queue.test_l2_draft_window_slot_executes_at_real_wall_clock',
    'test_native35_performance_queue.Window.test_p_draft_window_slot_executes_at_real_wall_clock',
    'test_native35_queue.Window.test_every_native35_draft_window_slot_executes_at_real_wall_clock',
    'test_native39_lifecycle_queue.Window.test_lifecycle_draft_window_slot_executes_at_real_wall_clock',
    'test_native39_trunk_queue.Window.test_trunk_draft_window_slot_executes_at_real_wall_clock',
    # Also holds native40's exact-derivation and budget asserts; they go with it.
    'test_native40_box_queue.Queue.test_exact_derivation_shell_and_executable_window',
    'test_native41_box_queue.Window.test_draft_window_slot_executes_at_real_wall_clock',
    'test_native41_progress_queue.Window.test_draft_window_slot_executes_at_real_wall_clock',
    # Asserts native43_boundary_gate.window_ok(time.time()) directly (no PY_SLOT); added after chain j (lead's go, 2 Nov).
    'test_native43_boundary_queue.QueueTests.test_window_uses_same_budget_real_wall_time_and_edges',
    'test_native43_lf_queue.Window.test_l_f_draft_window_slot_executes_at_real_wall_clock',
    'test_native43_lifecycle_queue.Window.test_lifecycle_draft_window_slot_executes_at_real_wall_clock',
)
MODULES = frozenset(x.split('.', 1)[0] for x in CLOCK_BOUND_FROZEN)


def _ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _ids(item)
        else:
            yield item


def strip(suite):
    """(suite without the listed ids, excluded ids); an id is matched on its last three parts: module, class, method."""
    kept, excluded = unittest.TestSuite(), []
    for test in _ids(suite):
        key = '.'.join(test.id().split('.')[-3:])
        if key in CLOCK_BOUND_FROZEN:
            excluded.append(key)
        else:
            kept.addTest(test)
    return kept, excluded


def run_modules(names, verbosity=1):
    sys.path.insert(0, str(HERE))
    suite = unittest.TestSuite()
    for name in names:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(name)))
    kept, excluded = strip(suite)
    print('clock-bound excluded: ' + (', '.join(sorted(excluded)) or 'none'), file=sys.stderr, flush=True)
    result = unittest.TextTestRunner(verbosity=verbosity).run(kept)
    return 0 if result.wasSuccessful() else 1


def main(argv):
    if argv[:2] == ['-v', '-m']:
        return run_modules(argv[2:], verbosity=2)
    if argv[:1] == ['-m']:
        return run_modules(argv[1:])
    assert len(argv) == 1 and argv[0].endswith('.py'), 'usage: clock_bound.py SUITE.py | [-v] -m MOD [MOD ...]'
    return run_modules([Path(argv[0]).stem])


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
