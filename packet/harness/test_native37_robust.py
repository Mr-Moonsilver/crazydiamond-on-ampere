"""Native37 robustness: typed native-admission refusals and the harness FULL return. CPU only.

T1/T2 (the pinned-image scheduler composition) live in e3_native37_admission_probe.py and
run in the container self-run.  This suite holds the hub-side checks:

* PlannerTrace: the search receipt never changes staged_chunk_plan's result, and names the
  forced stop of a refused prompt (197315 -> stop 197296, gap 18, query bound exhausted).
* SessionRefusal: the real NativeOverlapSession, NativeControlClient and NativeSegmentDriver
  (test_e3_native_overlap's cold stack) on an armed split route.  A refused plan is held by
  the stock seam, exported in status, and prefill_end returns the typed refusal although the
  refused window journals nothing (the driver refuses that evidence window).  Without a
  refusal the same evidence error is raised unchanged (counterfactual).  leave_full ends
  the hold.
* RefusalDeltas: native_admission_refusal over two admission records.
* HarnessCleanup (T3): the exact W2 round argv through real main(argv), the real
  NativeRuntimeClient/LoopbackControl and the declared loopback APIServer leaf.  prefill_end
  answering 500, a typed refusal, or a dead peer: pause and leave_full follow, each in its own
  try; the lane error, completeness and first_error are the original ones.  Set
  NATIVE37_T3_HARNESS=<path to an e3_native_performance.py> to run the same assertions on
  another harness (the held-c negative control); unset, the tree's module is the subject.

Fixture leaves: native37_fixture_leaves.json.
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'patches/s5_engine')]
import engine_driver as ed  # noqa: E402
import native_overlap_control as nc  # noqa: E402
import native_overlap_session as ns  # noqa: E402
import scheduler_hooks as sh  # noqa: E402
import e3_native35_performance as rounds  # noqa: E402
import e3_native_performance as tree_perf  # noqa: E402
import test_e3_native_overlap as native  # noqa: E402
import test_native35_performance as w2  # noqa: E402

BLOCK, HASH, CAP, MIN_ROWS = 4048, 176, 1024, 64


def geometry_split(n):
    """Declared leaf for the pinned Scheduler._mamba_block_aligned_split at production
    geometry (block 4048, hash 176, budget 1024, partial mamba cache hits).  T1 in the
    container checks the band rule it implies against the pinned method itself."""
    def split(start, limit):
        prefill_end = n
        if start >= prefill_end:
            return limit
        lcp = n - n % BLOCK
        end = start + limit
        if end < prefill_end:
            aligned = end // BLOCK * BLOCK
            if aligned > start or BLOCK <= CAP:
                end = aligned
        next_block = (start // BLOCK + 1) * BLOCK
        tail = n // HASH * HASH
        stops = (next_block if start % BLOCK else 0, lcp, tail if lcp < tail < n else 0)
        end = min((s for s in stops if start < s < end), default=end)
        return max(end - start, 0)
    return split


class PlannerTrace(unittest.TestCase):
    def test_trace_never_changes_the_plan_and_names_the_forced_stop(self):
        for n in (4167, 98312, 197315, 201306, 49458, 48578, 194305):
            with self.subTest(n=n):
                trace = {}
                plain = sh.staged_chunk_plan(0, n - 1, CAP, MIN_ROWS, geometry_split(n))
                traced = sh.staged_chunk_plan(0, n - 1, CAP, MIN_ROWS, geometry_split(n), trace=trace)
                self.assertEqual(plain, traced)
                self.assertEqual(set(trace), {'queries', 'exhausted', 'furthest_boundary'})
                if traced:
                    self.assertEqual(trace['furthest_boundary'], n - 1)
                    self.assertFalse(trace['exhausted'])
        trace = {}
        self.assertEqual(sh.staged_chunk_plan(0, 197314, CAP, MIN_ROWS, geometry_split(197315), trace=trace), ())
        self.assertEqual(trace, {'queries': 65536, 'exhausted': True, 'furthest_boundary': 197296})
        trace = {}
        self.assertEqual(sh.staged_chunk_plan(0, 11, 4, MIN_ROWS, lambda s, n: n, trace=trace), ())
        self.assertEqual(trace, {'queries': 0, 'exhausted': False, 'furthest_boundary': 0})
        with self.assertRaises(sh.E3ContractError):
            sh.staged_chunk_plan(0, 11, 4, 1, lambda s, n: n, trace=[])


class RefusalDeltas(unittest.TestCase):
    KEYS = ('refused', 'alloc_failures', 'admitted', 'plan_searches', 'plan_memo_hits',
            'split_stock_deferrals', 'refusal_seq')

    def record(self, last=None, **counts):
        row = {key: 0 for key in self.KEYS}
        row.update(counts, schema='s5-native-admission/1', last_refusal=last)
        return row

    def test_window_deltas_type_the_refusal_and_nothing_else(self):
        last = {'reason': 'no-staged-plan', 'gap': 18, 'furthest_boundary': 197296}
        begin = self.record(refused=2, refusal_seq=2, admitted=5)
        self.assertIsNone(ns.native_admission_refusal(None, begin))  # an older engine exports nothing
        self.assertIsNone(ns.native_admission_refusal(begin, dict(begin)))
        self.assertIsNone(ns.native_admission_refusal(begin, self.record(refused=2, refusal_seq=2, admitted=6,
                                                                         split_stock_deferrals=3)))
        typed = ns.native_admission_refusal(begin, self.record(last, refused=9, refusal_seq=9, admitted=5,
                                                               plan_searches=1, plan_memo_hits=6,
                                                               split_stock_deferrals=7))
        self.assertEqual(typed['reason'], 'native-admission-refused:no-staged-plan')
        self.assertEqual(typed['deltas'], {'refused': 7, 'alloc_failures': 0, 'admitted': 0, 'plan_searches': 1,
                                           'plan_memo_hits': 6, 'split_stock_deferrals': 7, 'refusal_seq': 7})
        self.assertEqual(typed['last_refusal'], last)
        held = ns.native_admission_refusal(begin, self.record(refused=2, refusal_seq=2, admitted=5,
                                                              split_stock_deferrals=4))
        self.assertEqual(held['reason'], 'native-admission-refused:stock-deferred-without-admission')
        self.assertEqual(ns.native_admission_refusal(begin, self.record(refused=1))['reason'],
                         'native-admission-record-invalid')
        alloc = ns.native_admission_refusal(begin, self.record({'reason': 'allocation-failed'}, refused=2,
                                                               refusal_seq=3, admitted=5, alloc_failures=1))
        self.assertEqual(alloc['reason'], 'native-admission-refused:allocation-failed')


class SessionRefusal(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stack = native.cold_stack()
        self.core, self.frontend = native.core_for(self.stack)
        self.pump = native.Pump(self.core).start()
        self.control = nc.NativeControlClient(self.frontend, timeout_s=5)
        self.session = ns.NativeOverlapSession(self.frontend, control=self.control)

    async def asyncTearDown(self):
        await self.pump.stop()
        if self.pump.failure:
            raise self.pump.failure

    async def call(self, action, **fields):
        return await self.session.execute(dict(schema=1, action=action, run_id='native37-session', **fields))

    async def test_refused_plan_is_held_typed_and_released_by_leave_full(self):
        scheduler, controller = self.stack.scheduler, self.stack.controller
        await self.call('prepare', target_route='split', phase='split-prefill-solo')
        begin = await self.call('prefill_begin', phase='split-prefill-solo')
        self.assertEqual(begin['snapshot']['native_admission']['schema'], 's5-native-admission/1')
        # The fixture's 12-token request at chunk 4 cannot plan at the production
        # minimum of 64 staged rows (cap 4 < 64): a typed no-staged-plan.
        controller.minimum_stage_rows = sh.QSA_STAGE_MIN_ROWS
        request = scheduler.waiting.peek_request()
        for _ in range(3):
            self.assertIsNone(sh.flash_s5_admit(scheduler))
            self.assertTrue(sh.defer_stock_prefix_hit(scheduler, None))
        self.assertIs(scheduler.waiting.peek_request(), request)
        self.assertIsNone(controller.parked)
        status = self.stack.driver.submit_control({'action': 'status'})['native_admission']
        self.assertEqual((status['plan_searches'], status['plan_memo_hits'], status['split_stock_deferrals']),
                         (1, 2, 3))
        self.assertEqual(status['last_refusal']['reason'], 'no-staged-plan')
        self.assertEqual((status['last_refusal']['furthest_boundary'], status['last_refusal']['gap']), (0, 11))
        end = await self.call('prefill_end')
        self.assertFalse(end['complete'])
        self.assertEqual(end['refusal']['reason'], 'native-admission-refused:no-staged-plan')
        self.assertEqual(end['refusal']['deltas']['refused'], 3)
        self.assertIn('overlap-journal-empty-window', end['evidence_error']['message'])
        self.assertIsNone(self.session.measure_start_ns)
        returned = await self.call('leave_full')
        self.assertEqual(returned['route'], 'full')
        self.assertIsNone(self.stack.driver.admission_intent())
        self.assertFalse(sh.defer_stock_prefix_hit(scheduler, None))
        self.assertEqual(controller.split_stock_deferrals, 3)

    async def test_without_a_refusal_the_evidence_error_is_raised_unchanged(self):
        await self.call('prepare', target_route='split', phase='split-prefill-solo')
        await self.call('prefill_begin', phase='split-prefill-solo')
        with self.assertRaises(Exception) as caught:
            await self.call('prefill_end')
        self.assertIn('overlap-journal-empty-window', str(caught.exception))
        self.assertIsNotNone(self.session.measure_start_ns)  # unchanged: the window stays open

    async def test_full_is_stock_but_unarmed_split_still_holds(self):
        self.assertFalse(sh.defer_stock_prefix_hit(self.stack.scheduler, None))
        await self.call('prepare', target_route='split')  # split route, no admission phase
        self.assertIsNone(self.stack.driver.admission_intent())
        self.assertTrue(sh.defer_stock_prefix_hit(self.stack.scheduler, None))
        self.assertEqual(self.stack.controller.split_stock_deferrals, 1)


def t3_subject():
    """The harness module under test: the tree's, or NATIVE37_T3_HARNESS (negative control)."""
    path = os.environ.get('NATIVE37_T3_HARNESS')
    if not path:
        return tree_perf
    spec = importlib.util.spec_from_file_location('native37_t3_subject', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EndVariantServer(w2.LoopbackAPIServer):
    """The declared loopback APIServer leaf, with prefill_begin accepted and prefill_end varied."""

    def __init__(self, end):
        self.end = end
        super().__init__()

    def session(self, handler, command):
        action = command['action']
        if action not in ('prefill_begin', 'prefill_end'):
            return super().session(handler, command)
        with self.lock:
            self.commands.append(command)
        reply = {'schema': 1, 'action': action, 'run_id': command['run_id'], 'engine_epoch': w2.EPOCH}
        if action == 'prefill_begin':
            reply.update(route='split', phase=command.get('phase'))
            return handler.reply(200, reply)
        if self.end == '500':
            return handler.reply(500, {'error': 'engine-dead'})
        if self.end == 'refusal':
            reply.update(complete=False, refusal={'reason': 'native-admission-refused:no-staged-plan',
                                                  'deltas': {'refused': 1}, 'last_refusal': {'gap': 18}})
            return handler.reply(200, reply)
        # Dead peer: the listener is closed before this request is dropped unanswered,
        # so every later connection is refused (deterministic, no reset race).
        self.close()
        handler.close_connection = True
        return None

    def close(self):
        with self.lock:
            if getattr(self, '_closed', False):
                return
            self._closed = True
        super().close()


class HarnessCleanup(unittest.TestCase):
    CASES = {
        '500': dict(error='http-status-500', cleanup_error=None, route='full', sent=11, retained=10, seen=11),
        'refusal': dict(error='native-admission-refused:no-staged-plan', cleanup_error=None, route='full',
                        sent=11, retained=11, seen=11),
        'dead-peer': dict(error='transport-failed:RemoteDisconnected',
                          cleanup_error='transport-failed:ConnectionRefusedError', route=None,
                          sent=11, retained=8, seen=9),
    }
    FULL = ['prepare', 'pause', 'leave_full', 'prepare', 'pause', 'leave_full']

    def run_round(self, end):
        server = EndVariantServer(end)
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            self.assertEqual(w2.write_gates(directory), 0)
            (directory / 'w1.json').write_text(json.dumps(
                {'schema': 2, 'mode': 'native-overlap', 'summary': {'verdict': 'INCONCLUSIVE', 'routes': {}}, 'repeats': []}))
            extra, argv = w2.w2_argv(directory, server.url, 0)
            with w2.package_inventory(), patch.dict(os.environ, w2.round_env(extra)), patch.object(rounds, 'perf', t3_subject()):
                self.assertEqual(rounds.main(argv), 2)
            record = json.loads((directory / 'overlap-round-0.json').read_text())
        return record, [c['action'] for c in server.commands]

    def test_lane_errors_are_kept_and_the_route_is_returned_best_effort(self):
        for end, want in self.CASES.items():
            with self.subTest(end=end):
                record, actions = self.run_round(end)
                lane = record['lanes']['prefill_split']['fresh']
                self.assertEqual(lane['error']['error_message'], want['error'])
                self.assertFalse(lane['complete'])
                self.assertEqual(record['first_error']['error_message'], 'native-lane-refused:prefill_split:fresh')
                self.assertEqual(record['lanes']['prefill_full']['fresh']['complete'], True)
                tail = ['prepare', 'prefill_begin', 'prefill_end']
                self.assertEqual(actions, self.FULL + tail + (['pause', 'leave_full'] if want['seen'] == 11 else []))
                cleanup = lane.get('cleanup')
                self.assertIsInstance(cleanup, dict, 'no FULL return after the lane error')
                self.assertTrue(cleanup['attempted'])
                for step in ('pause', 'leave_full'):
                    error = cleanup[step + '_error']
                    self.assertEqual(None if error is None else error['error_message'], want['cleanup_error'])
                self.assertEqual(cleanup['route_after'], want['route'])
                meta = record['native35_round']['session_client']
                self.assertEqual((meta['commands_sent'], meta['replies_retained']), (want['sent'], want['retained']))
                self.assertEqual(record['grade']['verdict'], 'INCONCLUSIVE')


if __name__ == '__main__':
    unittest.main()
