"""Native39 trunk: the Codex native37 review's P1/P2 robustness gaps. CPU only.

Each class names the finding it closes and fails on the 72ea0ba engine:

* GeneratedSeams (P1 #1): the generated scheduler holds a positive-frontier
  waiting request (KV receive, streaming resume, a native request between
  chunks) and an in-progress RUNNING stock prefill chunk on an armed split
  route, in the exact branches the pinned schedule() takes for them, with an
  exact revert.  The pinned-image probe (e3_native39_trunk_probe.py) drives
  both branches through the real AsyncScheduler.
* HeldKinds (P1 #1): the seams hold only while a native intent is armed,
  count per kind, and the native one-token handback is never held: from the
  terminal commit to the FULL ack the driver skips every stock step, and the
  ack closes the intent.
* RetainedRow (P1 #1 liveness): a held or refused request ahead of the
  retained native request no longer strands its worker row.
* RefusalMemo (P2 #3): the refused-plan memo never keeps a finished request
  alive and forgets it on the generated free seam.
* TypedRefusals (P2 #4, #6): full-prefix-hit and resumed-with-output are typed
  refusals; no-staged-plan names how its furthest boundary was derived.
* HoldBound (P2 #4): an idle armed-split hold ends on its own after a bound by
  returning FULL with a typed 'split-hold-timeout'; never before the bound,
  never while native work is in flight, never after the held requests left.
* EvidenceFallback (P2 #5): prefill_end types only the empty-window evidence
  refusal, only for a receipt observed inside the window on the same engine.

Real objects: PrefillAdmissionController, SchedulerLeases, NativeSegmentDriver,
the R4 worker bridges and native route objects (test_e3_native_overlap's cold
stack), NativeControlClient, NativeOverlapSession and the generated scheduler
source.  Fixture leaves: native39_fixture_leaves.json.
"""
import ast
import asyncio
import gc
import hashlib
from pathlib import Path
import sys
import time
import unittest
from unittest import mock
import weakref

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'patches/s5_engine')]
import engine_driver as ed  # noqa: E402
import mirror_hooks  # noqa: E402
import native_overlap_control as nc  # noqa: E402
import native_overlap_session as ns  # noqa: E402
import scheduler_hooks as sh  # noqa: E402
import telemetry_hooks  # noqa: E402
import test_e3_admission_native as adm  # noqa: E402
import test_e3_native_overlap as native  # noqa: E402
import test_e3_native_storage_replay as replay  # noqa: E402


def e2_scheduler_source():
    source = (ROOT / 'reference/vllm-src' / sh.SCHEDULER_PATH).read_text()
    return mirror_hooks.scheduler(telemetry_hooks.scheduler(source))


def hook_calls(nodes, name):
    count = 0
    for node in nodes:
        for item in ast.walk(node):
            if (isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                    and item.func.attr == name and ast.unparse(item.func.value) == '_flash_s5_hooks'):
                count += 1
    return count


class IterQueue(adm.Queue):
    """Leaf: the pinned FCFSRequestQueue (a deque) is iterable; the fake is not."""

    def __iter__(self):
        return iter(list(self.items))


def fcfs(scheduler):
    """Leaf: pinned Scheduler._select_waiting_queue_for_scheduling under FCFS."""
    scheduler._select_waiting_queue_for_scheduling = (
        lambda: scheduler.skipped_waiting or scheduler.waiting or None)


async def settle(predicate, *, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('engine ticks did not reach the expected state')
        await asyncio.sleep(0)


class GeneratedSeams(unittest.TestCase):
    def test_both_positive_frontier_branches_carry_a_seam_and_revert_exactly(self):
        e2 = e2_scheduler_source()
        self.assertEqual(hashlib.sha256(e2.encode()).hexdigest(), sh.SCHEDULER_E2_SHA256)
        out = sh.generate_scheduler(e2)
        self.assertEqual(sh.revert_scheduler(out), e2)
        schedule = [node for node in ast.walk(ast.parse(out))
                    if isinstance(node, ast.FunctionDef) and node.name == 'schedule']
        self.assertEqual(len(schedule), 1)
        schedule = schedule[0]
        # Waiting loop: the fresh branch keeps E1, the positive-frontier branch
        # (pinned scheduler.py:1145, KV receive / streaming resume) gets its own.
        fresh = [node for node in ast.walk(schedule) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == 'request.num_computed_tokens == 0']
        self.assertEqual(len(fresh), 1)
        fresh = fresh[0]
        self.assertEqual(hook_calls(fresh.body, 'defer_stock_prefix_hit'), 1)
        self.assertEqual(hook_calls(fresh.body, 'defer_stock_positive_frontier'), 0)
        self.assertEqual(hook_calls(fresh.orelse, 'defer_stock_positive_frontier'), 1)
        self.assertEqual(hook_calls(fresh.orelse, 'defer_stock_prefix_hit'), 0)
        held = [node for node in fresh.orelse if isinstance(node, ast.If)
                and hook_calls([node.test], 'defer_stock_positive_frontier')]
        self.assertEqual(len(held), 1)
        # The same disposition as E1: off this step's queue, kept waiting.
        self.assertEqual([ast.unparse(s) for s in held[0].body],
                         ['request_queue.pop_request()',
                          'step_skipped_waiting.prepend_request(request)', 'continue'])
        # Running loop (pinned scheduler.py:719): an in-progress prefill chunk.
        running = [node for node in ast.walk(schedule) if isinstance(node, ast.While)
                   and ast.unparse(node.test).startswith('req_index < len(self.running)')]
        self.assertEqual(len(running), 1)
        guard = [node for node in running[0].body if isinstance(node, ast.If)
                 and hook_calls([node.test], 'defer_stock_running_prefill')]
        self.assertEqual(len(guard), 1)
        self.assertIn('request.is_prefill_chunk', ast.unparse(guard[0].test))
        self.assertEqual([ast.unparse(s) for s in guard[0].body], ['req_index += 1', 'continue'])
        self.assertEqual(hook_calls([schedule], 'defer_stock_running_prefill'), 1)
        self.assertEqual(hook_calls([schedule], 'defer_stock_positive_frontier'), 1)


class Session(unittest.IsolatedAsyncioTestCase):
    """The real NativeOverlapSession/NativeControlClient over the cold stack."""

    async def asyncSetUp(self):
        self.stack = native.cold_stack()
        self.core, self.frontend = native.core_for(self.stack)
        self.pump = native.Pump(self.core).start()
        self.control = nc.NativeControlClient(self.frontend, timeout_s=5)
        self.session = ns.NativeOverlapSession(self.frontend, control=self.control)
        self.scheduler, self.controller, self.driver = (
            self.stack.scheduler, self.stack.controller, self.stack.driver)
        self.request = self.scheduler.waiting.peek_request()

    async def asyncTearDown(self):
        await self.pump.stop()
        if self.pump.failure:
            raise self.pump.failure

    async def call(self, action, **fields):
        return await self.session.execute(dict(schema=1, action=action, run_id='native39-session', **fields))

    async def arm(self):
        await self.call('prepare', target_route='split', phase='split-prefill-solo')
        return await self.call('prefill_begin', phase='split-prefill-solo')

    def schedule_pass(self, *requests):
        """One schedule() call's E3 seams: admission, then the stock holds."""
        record = sh.flash_s5_admit(self.scheduler)
        held = []
        for kind, request in requests:
            if kind == 'fresh':
                held.append(sh.defer_stock_prefix_hit(self.scheduler, None))
            elif kind == 'positive-frontier':
                held.append(sh.defer_stock_positive_frontier(self.scheduler, request))
            else:
                held.append(sh.defer_stock_running_prefill(self.scheduler, request))
        return record, held

    def status(self):
        return self.driver.submit_control({'action': 'status'})['native_admission']

    def kinds(self):
        return [entry['kind'] for entry in self.driver._journal]


class HeldKinds(Session):
    async def test_positive_frontier_and_running_prefill_hold_only_while_armed(self):
        running = adm.Request(uid='req-n39-running', tokens=40, computed=8)
        resumed = adm.Request(uid='req-n39-resumed', tokens=40, computed=8)
        self.assertFalse(sh.defer_stock_positive_frontier(self.scheduler, resumed))
        self.assertFalse(sh.defer_stock_running_prefill(self.scheduler, running))
        await self.call('prepare', target_route='split')  # split route, no admission intent
        self.assertFalse(sh.defer_stock_positive_frontier(self.scheduler, resumed))
        self.assertFalse(sh.defer_stock_running_prefill(self.scheduler, running))
        self.assertEqual(self.controller.split_stock_deferrals, 0)

        armed = await self.call('prepare', target_route='split', phase='split-prefill-solo')
        self.assertEqual(armed['route'], 'split')
        self.controller.minimum_stage_rows = sh.QSA_STAGE_MIN_ROWS  # the fresh request refuses
        for _ in range(2):
            record, held = self.schedule_pass(('fresh', self.request), ('positive-frontier', resumed),
                                              ('running-prefill', running))
            self.assertIsNone(record)
            self.assertEqual(held, [True, True, True])
        status = self.status()
        self.assertEqual(status['split_stock_deferrals'], 6)
        self.assertEqual(status['split_stock_deferrals_by_kind'],
                         {'fresh': 2, 'positive-frontier': 2, 'running-prefill': 2})
        self.assertEqual(status['split_hold']['holds'], 6)
        self.assertEqual(status['split_hold']['schedule_seq'], 2)
        self.assertEqual(status['split_hold_bound_ns'], sh.SPLIT_HOLD_BOUND_NS)
        self.assertEqual(status['split_hold_timeouts'], 0)
        returned = await self.call('leave_full')
        self.assertEqual(returned['route'], 'full')
        self.assertFalse(sh.defer_stock_positive_frontier(self.scheduler, resumed))
        self.assertFalse(sh.defer_stock_running_prefill(self.scheduler, running))
        self.assertEqual(self.controller.split_stock_deferrals, 6)

    async def test_the_native_one_token_handback_is_never_held(self):
        await self.pump.stop()  # this test owns every tick
        task = asyncio.get_running_loop().create_task(self.control.enter_split())
        await native.pump_until(lambda: task.done(), self.core)
        await self.control.arm_admission('split-prefill-solo')
        self.controller.split_hold_bound_ns = 0  # native work in flight is never timed out
        request, n = self.request, self.request.num_tokens
        stock_ticks = []
        chunks = []
        while True:
            record, held = self.schedule_pass(('positive-frontier', request))
            if record is None:
                break
            # While native work is in flight nothing stock-owned is timed out.
            self.assertIsNone(self.controller._split_hold)
            chunks.append((record.transaction.chunk_start, record.transaction.chunk_end))
            self.stack.drive_to_finalize()
            self.stack.complete_everything()
            for _ in range(400):
                if self.driver.advance() is None:
                    # The engine would run schedule() now: stock may run.
                    stock_ticks.append((request.num_computed_tokens, self.driver.admission_intent()))
                if self.driver.tx is None and self.driver._return_pending is None:
                    break
        self.assertEqual(chunks, [(0, 4), (4, 8), (8, n - 1)])
        self.assertIsNone(self.driver.error)
        # From the terminal commit (n-1) to the FULL ack no stock step ran, and
        # the ack closed the intent: the handback reaches stock on FULL only.
        self.assertFalse([tick for tick in stock_ticks if tick[0] == n - 1 and tick[1] is not None])
        self.assertIsNone(self.driver.admission_intent())
        self.assertFalse(self.driver._split_ready)
        self.assertEqual(request.num_computed_tokens, n - 1)
        self.assertIs(self.scheduler.waiting.peek_request(), request)
        self.assertFalse(sh.defer_stock_positive_frontier(self.scheduler, request))
        closed = [entry for entry in self.driver._journal if entry['kind'] == 'admission_disarmed']
        self.assertEqual([entry.get('reason') for entry in closed], ['route-returned-full'])
        self.assertNotIn('admission_hold_timeout', self.kinds())
        self.assertEqual(self.controller.split_hold_timeouts, 0)


class RetainedRow(unittest.TestCase):
    def retained_stack(self):
        stack = replay.make_native_stack()
        stack.drive_to_finalize()
        stack.complete_everything()
        stack.drive(lambda: stack.driver.tx is None)
        retained = stack.driver.retained_request()
        self.assertIsNotNone(retained)
        request = stack.scheduler.waiting.items[0]
        self.assertEqual((request.request_id, request.num_computed_tokens), (retained.request_uid, 4))
        return stack, request

    def test_a_held_head_in_skipped_waiting_does_not_strand_the_retained_row(self):
        stack, request = self.retained_stack()
        scheduler = stack.scheduler
        held = adm.Request(uid='req-n39-held-head', tokens=12)
        scheduler.waiting = IterQueue([request])
        scheduler.skipped_waiting = IterQueue([held])  # E1 held it on an earlier step
        fcfs(scheduler)
        record = stack.controller.admit(scheduler)
        self.assertIsNotNone(record)
        self.assertEqual((record.transaction.request_uid, record.transaction.chunk_start,
                          record.transaction.chunk_end), (request.request_id, 4, 8))
        self.assertIs(stack.controller.parked.request, request)
        self.assertIs(stack.controller.parked.queue, scheduler.waiting)
        self.assertEqual(scheduler.waiting.items, [])
        self.assertEqual(scheduler.skipped_waiting.items, [held])  # untouched, still first

    def test_a_refused_head_in_the_same_queue_does_not_strand_it_either(self):
        stack, request = self.retained_stack()
        scheduler = stack.scheduler
        head = adm.Request(uid='req-n39-refused-head', tokens=12)
        scheduler.waiting = IterQueue([head, request])
        record = stack.controller.admit(scheduler)
        self.assertEqual((record.transaction.request_uid, record.transaction.chunk_start),
                         (request.request_id, 4))
        self.assertEqual(scheduler.waiting.items, [head])

    def test_only_the_exact_retained_object_continues(self):
        stack, request = self.retained_stack()
        scheduler = stack.scheduler
        impostor = adm.Request(uid=request.request_id, tokens=12, computed=4)
        scheduler.waiting = IterQueue([impostor])
        self.assertIsNone(stack.controller.admit(scheduler))
        self.assertIsNone(stack.controller.parked)
        self.assertEqual(scheduler.waiting.items, [impostor])


class RefusalMemo(Session):
    async def test_a_freed_request_leaves_the_memo(self):
        await self.arm()
        self.controller.minimum_stage_rows = sh.QSA_STAGE_MIN_ROWS
        self.assertEqual(self.schedule_pass(('fresh', self.request)), (None, [True]))
        self.assertIn(self.request.request_id, self.controller._plan_refusals)
        # The generated free seam (scheduler._free_request -> defer_request_free).
        self.scheduler.waiting.remove_request(self.request)
        sh.defer_request_free(self.scheduler, self.request)
        self.assertNotIn(self.request.request_id, self.controller._plan_refusals)

    async def test_the_memo_never_keeps_a_finished_request_alive(self):
        await self.arm()
        self.controller.minimum_stage_rows = sh.QSA_STAGE_MIN_ROWS
        gone = adm.Request(uid='req-n39-memo', tokens=12)
        self.scheduler.waiting.prepend_request(gone)
        self.assertEqual(self.schedule_pass(('fresh', gone)), (None, [True]))
        self.assertIn('req-n39-memo', self.controller._plan_refusals)
        probe = weakref.ref(gone)
        self.scheduler.waiting.remove_request(gone)
        del gone
        gc.collect()
        self.assertIsNone(probe())  # nothing in the engine keeps it alive
        self.assertIsNone(self.controller._memoised_plan_refusal(self.request, None))
        # The next insert prunes the dead entry.
        self.assertEqual(self.schedule_pass(('fresh', self.request)), (None, [True]))
        self.assertNotIn('req-n39-memo', self.controller._plan_refusals)
        self.assertIn(self.request.request_id, self.controller._plan_refusals)


class TypedRefusals(Session):
    async def test_full_prefix_hit_and_resumed_output_are_typed_and_held(self):
        await self.arm()
        manager = self.scheduler.kv_cache_manager
        allocs = len(manager.alloc_calls)  # the cold stack's cancelled initial park
        manager.cached = self.request.num_tokens - 1
        self.assertEqual(self.schedule_pass(('fresh', self.request)), (None, [True]))
        last = self.status()['last_refusal']
        self.assertEqual((last['reason'], last['prompt_tokens'], last['cached_tokens']),
                         ('full-prefix-hit', 12, 11))
        self.assertEqual(last['request_uid'], self.request.request_id)
        self.assertIsNone(self.controller.parked)
        manager.cached = 0
        self.request.num_output_tokens = 2  # preempted after output: recomputes prompt + output
        self.assertEqual(self.schedule_pass(('fresh', self.request)), (None, [True]))
        last = self.status()['last_refusal']
        self.assertEqual((last['reason'], last['start'], last['output_tokens']),
                         ('resumed-with-output', 0, 2))
        self.assertEqual(len(manager.alloc_calls), allocs)

    async def test_no_staged_plan_declares_its_furthest_boundary_basis(self):
        await self.arm()
        self.controller.minimum_stage_rows = sh.QSA_STAGE_MIN_ROWS
        self.schedule_pass(('fresh', self.request))
        last = self.status()['last_refusal']
        self.assertEqual(last['reason'], 'no-staged-plan')
        self.assertEqual(last['furthest_basis'], sh.FURTHEST_BASIS)
        self.assertIn('not a certified stock-forced stop', last['furthest_basis'])


class HoldBound(Session):
    def register(self, *requests):
        # Leaf: pinned Scheduler.requests (every unfinished request by id).
        self.scheduler.requests = {request.request_id: request for request in requests}

    async def test_an_idle_hold_past_its_bound_returns_full_with_a_typed_receipt(self):
        await self.arm()
        self.controller.minimum_stage_rows = sh.QSA_STAGE_MIN_ROWS
        self.register(self.request)
        self.controller.split_hold_bound_ns = 0
        self.assertEqual(self.schedule_pass(('fresh', self.request)), (None, [True]))
        await settle(lambda: self.driver.admission_intent() is None)
        self.assertFalse(self.driver._split_ready)
        status = self.status()
        self.assertEqual(status['split_hold_timeouts'], 1)
        self.assertIsNone(status['split_hold'])
        last = status['last_refusal']
        self.assertEqual((last['reason'], last['action'], last['holds'], last['bound_ns']),
                         ('split-hold-timeout', 'return-full', 1, 0))
        self.assertEqual(last['kinds'], {'fresh': 1, 'positive-frontier': 0, 'running-prefill': 0})
        self.assertEqual(last['request_uid'], self.request.request_id)
        kinds = self.kinds()
        self.assertLess(kinds.index('admission_hold_timeout'), kinds.index('admission_disarmed'))
        disarmed = [entry for entry in self.driver._journal if entry['kind'] == 'admission_disarmed']
        self.assertEqual(disarmed[-1].get('reason'), 'route-returned-full')
        # Stock owns the request again, on FULL.
        self.assertFalse(sh.defer_stock_prefix_hit(self.scheduler, None))
        self.assertIs(self.scheduler.waiting.peek_request(), self.request)
        end = await self.call('prefill_end')
        self.assertFalse(end['complete'])
        self.assertEqual(end['refusal']['reason'], 'native-admission-refused:split-hold-timeout')
        self.assertEqual(end['refusal_policy'], ns.REFUSAL_POLICY)
        returned = await self.call('leave_full')
        self.assertEqual(returned['route'], 'full')

    async def test_a_plannable_follower_behind_a_refused_head_ends_in_the_bounded_return(self):
        await self.arm()
        admitted = self.controller.admitted
        follower = adm.Request(uid='req-n39-follower', tokens=12)
        self.scheduler.waiting.items.append(follower)
        self.request.num_output_tokens = 1  # the head refuses (resumed-with-output)
        self.register(self.request, follower)
        for _ in range(3):
            self.assertEqual(self.schedule_pass(('fresh', self.request)), (None, [True]))
        # FCFS: native admission reads the head only; the follower waits (it
        # is plannable: the same 12-token geometry plans at chunk 4).
        self.assertIsNone(self.controller.parked)
        self.assertEqual(self.controller.admitted, admitted)
        self.controller.split_hold_bound_ns = 0
        self.assertEqual(self.schedule_pass(('fresh', self.request)), (None, [True]))
        await settle(lambda: self.driver.admission_intent() is None)
        self.assertEqual(self.status()['last_refusal']['reason'], 'split-hold-timeout')
        self.assertEqual(self.status()['last_refusal']['holds'], 4)
        self.assertFalse(sh.defer_stock_prefix_hit(self.scheduler, None))

    async def test_no_return_before_the_bound_without_holds_or_after_the_requests_left(self):
        await self.arm()
        self.controller.minimum_stage_rows = sh.QSA_STAGE_MIN_ROWS
        self.register(self.request)
        intent = self.driver.admission_intent()
        # Before the bound.
        self.controller.split_hold_bound_ns = 3_600_000_000_000
        self.schedule_pass(('fresh', self.request))
        ticks = self.core.ticks
        await settle(lambda: self.core.ticks >= ticks + 200)
        self.assertEqual(self.driver.admission_intent(), intent)
        # A schedule() call that held nothing ends the hold (the held request
        # was scheduled elsewhere or left; decodes may still run).
        self.controller.split_hold_bound_ns = 0
        self.assertIsNone(sh.flash_s5_admit(self.scheduler))
        ticks = self.core.ticks
        await settle(lambda: self.core.ticks >= ticks + 200)
        self.assertEqual(self.driver.admission_intent(), intent)
        # Held again, then the client went away: nothing is stranded, so the
        # armed route stays as the harness set it.
        self.schedule_pass(('fresh', self.request))
        self.scheduler.requests = {}
        ticks = self.core.ticks
        await settle(lambda: self.core.ticks >= ticks + 200)
        self.assertEqual(self.driver.admission_intent(), intent)
        # Paused (a harness route change in progress): no return either.
        self.register(self.request)
        self.core.paused = True
        ticks = self.core.ticks
        await settle(lambda: self.core.ticks >= ticks + 200)
        self.assertEqual(self.driver.admission_intent(), intent)
        self.assertEqual(self.controller.split_hold_timeouts, 0)
        self.assertNotIn('admission_hold_timeout', self.kinds())
        # Unpaused with the hold still current: the bound applies.
        self.core.paused = False
        await settle(lambda: self.driver.admission_intent() is None)
        self.assertEqual(self.controller.split_hold_timeouts, 1)


class EvidenceFallback(Session):
    async def refused_window(self):
        await self.arm()
        self.controller.minimum_stage_rows = sh.QSA_STAGE_MIN_ROWS
        self.assertEqual(self.schedule_pass(('fresh', self.request)), (None, [True]))

    async def test_another_evidence_failure_is_raised_unchanged_despite_a_refusal(self):
        await self.refused_window()
        failure = nc.NativeControlError('native-control-evidence-identity-mismatch')
        with mock.patch.object(self.control, 'overlap_evidence', side_effect=failure):
            with self.assertRaises(nc.NativeControlError) as caught:
                await self.call('prefill_end')
        self.assertIs(caught.exception, failure)

    async def test_a_receipt_from_outside_the_window_is_not_attributed_to_it(self):
        await self.arm()
        self.controller.minimum_stage_rows = sh.QSA_STAGE_MIN_ROWS
        with mock.patch.object(self.controller, '_now_ns', return_value=1):
            self.assertEqual(self.schedule_pass(('fresh', self.request)), (None, [True]))
        self.assertEqual(self.status()['last_refusal']['observed_ns'], 1)
        with self.assertRaises(Exception) as caught:
            await self.call('prefill_end')
        self.assertIn(ns.EMPTY_WINDOW_ERROR, str(caught.exception))

    async def test_the_in_window_refusal_is_still_typed_with_its_policy(self):
        await self.refused_window()
        end = await self.call('prefill_end')
        self.assertFalse(end['complete'])
        self.assertEqual(end['refusal']['reason'], 'native-admission-refused:no-staged-plan')
        self.assertEqual(end['refusal_policy'], ns.REFUSAL_POLICY)
        self.assertIn(ns.EMPTY_WINDOW_ERROR, end['evidence_error']['message'])


if __name__ == '__main__':
    unittest.main()
