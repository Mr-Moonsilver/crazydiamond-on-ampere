"""Load-robust twin of test_native43_le_bounded_enqueue.BoundedEnqueueTests.
test_the_le_chain_raises_the_stored_worker_rpc_error_within_the_bound.

The frozen test bounds ``row.ended - sent`` by ``0.05 + 1.0`` s of wall clock, which times two things together:
the bounded-enqueue slice and the fatal-channel thread's hand-off.  Under load the hand-off thread can be
descheduled for more than a second.  Here the bound is exact in slices and needs no clock:

* fixture leaf ``SteppedRing`` replaces the frozen test's ``Ring`` (itself the stand-in for
  ``MessageQueue.acquire_write``'s timeout contract, native43_le_bounded_enqueue_fixture_leaves.json): each
  ``enqueue(obj, timeout)`` with a timeout blocks until the test releases that slice, then raises
  ``TimeoutError`` with nothing written; it counts its calls.  ``timeout=None`` waits for a free block as the
  stock call does (never, here), so a stock unbounded write hangs and is caught by the liveness deadline;
* sequence: release slice 1; the thread is alive and enters slice 2; ``publish_fatal``; wait until
  ``executor._flash_s5_fatal_failure is not None`` (the hand-off); note ``calls_at_handoff``; release the
  in-progress slice.  The sequence runs twice, with the fatal record published in slice 2 (PLAN 5b) and then in
  slice 3 (two live slices first): a mutant that looks at the stored failure only every N slices (any N >= 2,
  any phase) lines up with at most one of two consecutive hand-off slices, so one pass catches it exactly;
* asserts (each pass): the call ends within that slice (thread ended and ``len(ring.calls) == calls_at_handoff``, so no
  new slice starts after the hand-off is visible), ``row.error is executor._flash_s5_fatal_failure`` and is a
  ``WorkerRPCError`` with the l-e record, ``ring.writes == []``, ``len(executor.shutdowns) == 1``.

Dropped, declared: the frozen test's 1.0 s allowance for the fatal thread's hand-off (scheduler time, not the
bounded-enqueue contract).  Every wait has a 30 s liveness deadline; here a miss is a FAIL (each waited-for step
is product code: the bounded write, the fatal channel), with the ring's calls in the message.  The actual
executor_hooks.bounded_enqueue and the actual fatal channel (monitor_fatal_workers over a real pipe) run; the
executor namespace and publish_fatal are the frozen module's (its tests are not loaded from here).
Leaves: alpha_next_robust_fixture_leaves.json.
"""
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_native43_le_bounded_enqueue as frozen  # noqa: E402 - fixtures only (module object, no re-collection)

hooks = frozen.hooks
WorkerRPCError = frozen.WorkerRPCError
LIVENESS_S = 30.0


class SteppedRing(frozen.Ring):
    """Fixture leaf: every bounded slice is released by the test and then times out with nothing written."""

    def __init__(self):
        super().__init__(free=0)
        self.released = 0  # slices the test has let time out
        self.finished = False  # set by the call's thread when the call under test returns or raises

    def enqueue(self, obj, timeout=None):
        with self.cond:
            self.calls.append(timeout)
            slice_no = len(self.calls)
            self.cond.notify_all()
            if timeout is None:  # stock acquire_write: wait for a free block (the liveness bound only frees the thread)
                if not self.cond.wait_for(lambda: self.free > 0, 2 * LIVENESS_S):
                    raise RuntimeError('fixture: unbounded write released by the fixture liveness bound')
                self.free -= 1
                self.writes.append(obj)
                return
            if not self.cond.wait_for(lambda: self.released >= slice_no, 2 * LIVENESS_S):
                raise RuntimeError('fixture: slice %d never released' % slice_no)
            raise TimeoutError  # the slice expired before a block was claimed: nothing written

    def release(self, slice_no):
        with self.cond:
            self.released = max(self.released, slice_no)
            self.cond.notify_all()

    def wait_until(self, predicate, what):
        with self.cond:
            if not self.cond.wait_for(predicate, LIVENESS_S):
                raise AssertionError('liveness deadline (%.0f s) missed: %s; ring calls %r, released %d, finished %s'
                                     % (LIVENESS_S, what, self.calls, self.released, self.finished))

    def finish(self):
        with self.cond:
            self.finished = True
            self.cond.notify_all()


def background(ring, fn):
    """frozen.background without the clock: the end is signalled on the ring's condition instead."""
    row = SimpleNamespace(value=None, error=None)

    def run():
        try:
            row.value = fn()
        except BaseException as exc:  # noqa: BLE001 - the case inspects it
            row.error = exc
        finally:
            ring.finish()
    row.thread = threading.Thread(target=run, daemon=True)
    row.thread.start()
    return row


def poll(predicate, what, step_s=0.001):
    """Wait until predicate() for state product threads change without a seam to signal (liveness bound only)."""
    deadline = time.monotonic() + LIVENESS_S
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('liveness deadline (%.0f s) missed: %s' % (LIVENESS_S, what))
        time.sleep(step_s)


class SteppedBoundedEnqueueTests(unittest.TestCase):
    def test_the_le_chain_raises_the_stored_worker_rpc_error_within_the_bound(self):
        for live_slices in (1, 2):  # the fatal record lands in slice 2 (PLAN 5b), then in slice 3
            with self.subTest(handoff_slice=live_slices + 1):
                self._le_chain(live_slices)

    def _le_chain(self, live_slices):
        ring = SteppedRing()
        executor, writer = frozen.fatal_executor(ring, self)
        self.addCleanup(ring.drain)  # an unbounded (timeout=None) write is freed by a block, not a slice
        self.addCleanup(ring.release, 10 ** 9)  # never leave the call's thread parked on a slice
        row = background(ring, lambda: hooks.bounded_enqueue(executor, ring, 'step', slice_s=0.05))
        ring.wait_until(lambda: len(ring.calls) >= 1, 'the bounded write enters slice 1')
        for slice_no in range(1, live_slices + 1):
            ring.release(slice_no)
            ring.wait_until(lambda: len(ring.calls) > slice_no or ring.finished,
                            'the bounded write enters slice %d' % (slice_no + 1))
            self.assertFalse(ring.finished, 'the bounded write ended on a live executor: %r' % (row.error,))
            self.assertTrue(row.thread.is_alive())  # live executor, full ring: keeps waiting
        frozen.publish_fatal(writer)
        poll(lambda: getattr(executor, '_flash_s5_fatal_failure', None) is not None, 'the fatal channel hand-off')
        calls_at_handoff = len(ring.calls)
        self.assertEqual(calls_at_handoff, live_slices + 1)  # no slice is released while the hand-off lands
        ring.release(calls_at_handoff)
        ring.wait_until(lambda: ring.finished or len(ring.calls) > calls_at_handoff,
                        'the bounded write ends after the released slice')
        self.assertEqual(len(ring.calls), calls_at_handoff,
                         'a new slice started after the hand-off was visible: calls %r' % ring.calls)
        self.assertTrue(ring.finished)
        row.thread.join(LIVENESS_S)
        self.assertFalse(row.thread.is_alive(), 'bounded write still blocked after the fatal record')
        self.assertIsInstance(row.error, WorkerRPCError)
        self.assertIs(row.error, executor._flash_s5_fatal_failure)
        self.assertEqual(row.error.record['error_type'], 'QsaDriverError')
        self.assertEqual(ring.writes, [])  # a timed-out slice writes nothing
        poll(lambda: len(executor.shutdowns) >= 1, 'the fatal channel runs its teardown')
        self.assertEqual(len(executor.shutdowns), 1)  # the fatal channel's own teardown still runs, once

if __name__ == '__main__':
    unittest.main()
