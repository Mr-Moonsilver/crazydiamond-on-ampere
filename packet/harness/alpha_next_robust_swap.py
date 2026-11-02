#!/usr/bin/env python3
"""ROBUST_SWAP for alpha-next (PLAN section 5): two load-flaky frozen test methods, left out by id, each replaced by its
load-robust twin under a new name.

The frozen files stay byte-identical (queues have run them).  Each listed frozen method asserts its contract through a
wall-clock race the hub's load breaks (5a: a 20 ms sleep stagger decides the advance order; 5b: a 1.05 s bound covers the
fatal thread's hand-off).  Its twin asserts the same contract with no wall-clock dependence and catches every mutant the
frozen method catches (mutant-parity table in the alpha-next delivery note); the twin is its own selftest row.

Uses: `python3 alpha_next_robust_swap.py MODULE [MODULE ...]` runs those test modules minus the listed frozen ids and minus
step_b_clock_bound's declared real-clock window-slot ids (so a gate chain's focused list can run through this one runner);
`filtered(suite)` for a bundle probe.  Both refuse unless every listed id of the loaded modules was found and removed exactly
once, and unless each removed id's robust twin loads.
"""
import sys
import unittest

import step_b_clock_bound

REASON = ('load-flaky wall-clock race (PLAN 5); replaced by its load-robust twin %s, which asserts the same contract '
          'without wall clock and catches every mutant the frozen method catches')
ROBUST_SWAP = {
    'test_e3_w1_decode_driver.DriverTest.test_streams_advance_out_of_order_but_all_must_open':
        'test_e3_w1_decode_driver_ordered.OrderedOpenTest.test_streams_advance_out_of_order_but_all_must_open',
    'test_native43_le_bounded_enqueue.BoundedEnqueueTests.'
    'test_the_le_chain_raises_the_stored_worker_rpc_error_within_the_bound':
        'test_native43_le_bounded_enqueue_stepped.SteppedBoundedEnqueueTests.'
        'test_the_le_chain_raises_the_stored_worker_rpc_error_within_the_bound',
}
MODULES = sorted({name.split('.', 1)[0] for name in ROBUST_SWAP})
ROBUST_MODULES = sorted({name.split('.', 1)[0] for name in ROBUST_SWAP.values()})


def reason(frozen_id):
    return REASON % ROBUST_SWAP[frozen_id]


def _flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


def filtered(suite):
    """The suite without the swapped frozen ids, and the ids removed; refuses an id found twice or a twin that does not load."""
    tests = list(_flatten(suite))
    removed = sorted(t.id() for t in tests if t.id() in ROBUST_SWAP)
    assert len(removed) == len(set(removed)), ('swapped id loaded twice', removed)
    for name in removed:
        twin = list(_flatten(unittest.defaultTestLoader.loadTestsFromName(ROBUST_SWAP[name])))
        assert [t.id() for t in twin] == [ROBUST_SWAP[name]], ('robust twin does not load', name, [t.id() for t in twin])
    return unittest.TestSuite(t for t in tests if t.id() not in ROBUST_SWAP), removed


def expected_removed(modules):
    return sorted(name for name in ROBUST_SWAP if name.split('.', 1)[0] in modules)


def main(modules):
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite(loader.loadTestsFromName(name) for name in modules)
    suite, clock_removed = step_b_clock_bound.filtered(suite)
    assert clock_removed == step_b_clock_bound.expected_removed(modules), (
        'declared clock-bound ids not found exactly once', clock_removed)
    kept, removed = filtered(suite)
    assert removed == expected_removed(modules), ('declared swapped ids not found exactly once', removed)
    for name in clock_removed:
        print('clock-bound frozen, excluded (declared): %s: %s' % (name, step_b_clock_bound.CLOCK_BOUND_FROZEN[name]))
    for name in removed:
        print('load-flaky frozen, swapped (declared): %s: %s' % (name, reason(name)))
    result = unittest.TextTestRunner(verbosity=1).run(kept)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    assert len(sys.argv) > 1, 'usage: alpha_next_robust_swap.py MODULE [MODULE ...]'
    sys.exit(main([name[:-3] if name.endswith('.py') else name for name in sys.argv[1:]]))
