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

SUPERSEDED (alpha-next composition): the stage-1 anatomy FreezeProof method, pinned to its own parent tree; superseded by
tools/alpha_next_freeze.py, which pins that feature's record and limits the engine delta after it.

Coverage note: modules are loaded by name (as step_b_clock_bound does), not run as scripts.  test_e3_w1_decode_driver.py has
a first `if __name__ == '__main__': unittest.main()` at line 1508, before ReviewRegressionTests (5 tests) is defined, so its
script-mode selftest row ran 52 tests; through this runner the row runs 56 (52 - 1 swapped + those 5, all green).
test_native43_le_bounded_enqueue runs 9 (10 - 1 swapped).
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
# A frozen method whose proof is pinned to a parent tree the composition moved past: left out by id, superseded by the named
# proof, which the build and the gate chain run (no twin test; the superseding record carries the feature's own record by sha).
SUPERSEDED = {
    'test_stage1_handler_anatomy.FreezeProof.test_source_proof_and_negative_controls':
        'stage-1 anatomy freeze proof diffs the working tree against its own parent step-b 5f46c07, so any composed tree fails it by '
        'construction; superseded by tools/alpha_next_freeze.py (composition proof, the stage-1 record freeze-f708f082.json pinned '
        'by sha with its PASS, the post-freeze engine delta limited to the new module handler_anatomy.py)',
}
MODULES = sorted({name.split('.', 1)[0] for name in ROBUST_SWAP} | {name.split('.', 1)[0] for name in SUPERSEDED})
ROBUST_MODULES = sorted({name.split('.', 1)[0] for name in ROBUST_SWAP.values()})


def reason(frozen_id):
    return SUPERSEDED[frozen_id] if frozen_id in SUPERSEDED else REASON % ROBUST_SWAP[frozen_id]


def _flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


def filtered(suite):
    """The suite without the swapped frozen ids, and the ids removed; refuses an id found twice or a twin that does not load."""
    tests = list(_flatten(suite))
    gone = set(ROBUST_SWAP) | set(SUPERSEDED)
    removed = sorted(t.id() for t in tests if t.id() in gone)
    assert len(removed) == len(set(removed)), ('swapped id loaded twice', removed)
    for name in removed:
        if name in SUPERSEDED:
            continue
        twin = list(_flatten(unittest.defaultTestLoader.loadTestsFromName(ROBUST_SWAP[name])))
        assert [t.id() for t in twin] == [ROBUST_SWAP[name]], ('robust twin does not load', name, [t.id() for t in twin])
    return unittest.TestSuite(t for t in tests if t.id() not in gone), removed


def expected_removed(modules):
    return sorted(name for name in list(ROBUST_SWAP) + list(SUPERSEDED) if name.split('.', 1)[0] in modules)


class _WriterResetResult(unittest.TextTestResult):
    """After every test, the process receipt writer slot goes back to the parent's None (alpha-next PLAN 2).

    A test that runs s5_startup.install_engine or install_native_lane installs the process ReceiptWriter, which in a real engine or
    worker process lives for the process. In one shared runner process it would redirect every later module's receipts away from
    the logger their frozen captures read. Strict runs each suite in its own process and needs no reset.
    """

    def stopTest(self, test):
        super().stopTest(test)
        reset_receipt_writer(test.id())


WRITER_RESETS = []


def reset_receipt_writer(test_id=''):
    """Put the process receipt writer slot back to None (see _WriterResetResult); for any shared-process runner. True if it reset."""
    measurement = sys.modules.get('native_measurement')
    if measurement is not None and getattr(measurement, '_RECEIPT_WRITER', None) is not None:
        measurement._RECEIPT_WRITER = None
        WRITER_RESETS.append(test_id)
        return True
    return False


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
        kind = 'pinned-parent frozen, superseded (declared)' if name in SUPERSEDED else 'load-flaky frozen, swapped (declared)'
        print('%s: %s: %s' % (kind, name, reason(name)))
    result = unittest.TextTestRunner(verbosity=1, resultclass=_WriterResetResult).run(kept)
    print('receipt writer slot reset after %d tests (declared, PLAN 2)' % len(WRITER_RESETS))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    assert len(sys.argv) > 1, 'usage: alpha_next_robust_swap.py MODULE [MODULE ...]'
    sys.exit(main([name[:-3] if name.endswith('.py') else name for name in sys.argv[1:]]))
