"""Kept-SPLIT finite output-cut proof on the real Core/worker-order path; CPU only."""
from collections import deque
import hashlib
import inspect
import itertools
import os
from pathlib import Path
import subprocess
import threading
import textwrap
import time
import types
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import test_control_inwait as ci
import test_lane_o3 as lo3
import test_append_yield as ay
import test_tail_select as ts
import native_auto_arm as auto
import executor_hooks as eh
import s5_startup


def kept_rig():
    r, _ = lo3.o3_rig()
    lo3.run_native_to_terminal(r, r.bulk)
    r.tick(lambda: r.p.phase == 'tail')
    lo3.split_handback_step(r, r.bulk)
    ts.landed(r, r.bulk)
    lo3.first_decode_step(r, r.bulk)
    r.core.batch_queue = deque()
    r.p.old_pause = ci.Pause.UNPAUSED
    r.s.pause_state = ci.Pause.UNPAUSED
    r.p.control_inwait = auto.ControlInWait(r.p)
    return r


class CoreOrder(unittest.TestCase):
    finish = ci.RealCoreOrder.finish

    def make_rig(self):
        return kept_rig()

    def setUp(self):
        ci.RealCoreOrder.setUp(self)
        # Wrap the final scheduler method, after the test's stock output leaf is installed.
        self.r.p.control_inwait.enable_o3()

    def probe(self):
        row = ci.rpc.threaded(lambda: self.step(self.r.core))
        deadline = time.monotonic() + 2
        while not any(item[0] == 'flash_s5_native_overlap' for item in self.wire.writes):
            if time.monotonic() > deadline:
                self.wire.sample_gate.set()
                ci.rpc.join(self, row)
                self.fail('census not submitted behind the real sample')
            time.sleep(.001)
        self.assertFalse(self.applied)
        self.assertEqual(self.r.p.phase, 'control_o3_observe')
        self.assertEqual(len(self.r.p.control_inwait.cut), 1)
        query = next(item for item in self.wire.writes if item[0] == 'flash_s5_native_overlap')
        self.assertEqual(query[1][0]['action'], 'census')
        self.assertIsNone(self.r.d.admission_intent())
        return row

    def release(self, row):
        self.wire.sample_gate.set()
        _, errors = ci.rpc.join(self, row)
        self.assertEqual(errors, [])

    def drain_remaining(self):
        while self.r.core.batch_queue:
            future, output, execute = self.r.core.batch_queue.pop()
            value = future.result(2)
            self.r.s.update_from_output(output, value)

    def test_kept_census_arms_with_later_decode_still_queued_without_route_change(self):
        generation = self.r.d.route_generation
        self.release(self.probe())
        self.finish()
        self.assertEqual(self.r.p.phase, 'native')
        self.assertEqual(self.r.d.route_generation, generation)
        self.assertTrue(self.r.core.batch_queue, 'the queue must not have drained to arm')
        self.assertNotIn('flash_s5_transition', [item[0] for item in self.wire.writes])
        for order in self.wire.order:
            where = order.index('flash_s5_native_overlap')
            self.assertEqual(order[where-2:where], ['execute_model', 'sample_tokens'])
        stats = self.r.d._native_admission_record()['control_inwait']['transitions']['split-kept-entry']
        self.assertEqual((stats['cut_frames_issued'], stats['cut_frames_applied']), (1, 1))
        self.assertEqual((stats['armed'], stats['arms_with_later_frames']), (1, 1))
        self.assertEqual(stats['drains_kept']['typed-route-fence'], 0)
        self.drain_remaining()
        self.assertFalse(self.transport._calls)
        self.assertFalse(self.wire.errors)

    def test_finite_cut_waits_for_update_to_return_not_just_future_or_queue_pop(self):
        entered, leave = threading.Event(), threading.Event()

        def delayed_update():
            entered.set()
            if not leave.wait(3):
                raise TimeoutError('fixture scheduler output gate')
        self.on_update = delayed_update
        row = self.probe()
        self.wire.sample_gate.set()
        try:
            self.assertTrue(entered.wait(2))
            # The Core already popped the old future; it has not applied that output.
            self.assertEqual(len(self.r.p.control_inwait.cut), 1)
            self.assertIsNone(self.r.d.admission_intent())
        finally:
            leave.set()
            _, errors = ci.rpc.join(self, row)
        self.assertEqual(errors, [])
        self.finish()
        self.assertEqual(self.r.p.phase, 'native')
        self.drain_remaining()

    def test_changed_cohort_drains_before_full_fallback(self):
        self.on_update = lambda: self.r.s.running.clear()
        self.release(self.probe())
        self.finish()
        self.assertEqual(self.r.p.phase, 'idle')
        self.assertFalse(self.r.d._split_ready)
        self.assertIsNone(self.r.d.admission_intent())
        self.assertFalse(self.r.core.batch_queue)
        stats = self.r.p.control_inwait.snapshot()['transitions']['split-kept-entry']
        self.assertEqual((stats['closed'], stats['armed']), (1, 0))
        self.assertEqual(stats['drains_kept']['fallback-full-cut'], 1)

    def test_real_first_handback_output_is_required_after_the_cut(self):
        self.r.bulk.num_output_tokens = 0
        self.on_update = lambda: setattr(self.r.bulk, 'num_output_tokens', 1)
        self.release(self.probe())
        self.finish()
        self.assertEqual(self.r.p.phase, 'native')
        self.assertEqual(self.r.bulk.num_output_tokens, 1)
        self.drain_remaining()

    def test_deferred_grammar_sample_stays_in_worker_order(self):
        self.r.s.schedule = lambda throttle: NS(total_num_scheduled_tokens=1, pending_structured_output_tokens=True)
        self.release(self.probe())
        self.finish()
        self.assertEqual(self.r.p.phase, 'native')
        self.assertTrue(self.r.core.batch_queue)
        self.drain_remaining()
        self.assertFalse(self.transport._calls)

    def test_missing_handback_output_or_revoked_ledger_never_arms(self):
        self.r.bulk.num_output_tokens = 0
        self.release(self.probe())
        self.finish()
        self.assertEqual(self.r.p.phase, 'idle')
        self.assertEqual(self.r.p.o3_counts['refused'].get('handback-output-unapplied'), 1)
        self.assertIsNone(self.r.d.admission_intent())

    def test_same_uid_request_replacement_cannot_reuse_the_old_census(self):
        def replace():
            old = self.r.s.running[0]
            replacement = ay.Req(old.request_id, old.num_tokens, computed=old.num_computed_tokens)
            replacement.status = NS(name='RUNNING')
            replacement.num_output_tokens = 1
            self.r.s.running[:] = [replacement]
            self.r.s.requests[old.request_id] = replacement
        self.on_update = replace
        self.release(self.probe())
        self.finish()
        self.assertEqual(self.r.p.phase, 'idle')
        self.assertIsNone(self.r.d.admission_intent())
        self.assertEqual(self.r.p.o3_counts['refused'].get('selection-changed-after-census'), 1)

    def test_revoked_ledger_is_rechecked_after_captured_outputs_apply(self):
        self.on_update = lambda: setattr(self.r.p, 'o3_joined', None)
        self.release(self.probe())
        self.finish()
        self.assertEqual(self.r.p.phase, 'idle')
        self.assertEqual(self.r.p.o3_counts['refused'].get('ledger'), 1)
        self.assertIsNone(self.r.d.admission_intent())

    def test_cached_split_generation_is_rechecked_after_captured_outputs_apply(self):
        self.on_update = lambda: setattr(self.r.p, 'o3_split_generation', self.r.p.o3_split_generation + 1)
        self.release(self.probe())
        self.finish()
        self.assertEqual(self.r.p.phase, 'idle')
        self.assertEqual(self.r.p.o3_counts['refused'].get('generation'), 1)
        self.assertIsNone(self.r.d.admission_intent())

    def test_selected_request_abort_during_output_application_never_admits_it(self):
        def abort():
            selected = self.r.p._request()
            if selected is not None:
                self.r.s.requests.pop(selected.request_id)
        self.on_update = abort
        self.release(self.probe())
        self.finish()
        self.assertEqual(self.r.p.phase, 'idle')
        self.assertIsNone(self.r.d.admission_intent())

    def test_capture_cannot_be_overwritten_or_grown_past_two_outputs(self):
        row = self.probe()
        try:
            with self.assertRaisesRegex(Exception, 'already-owned'):
                self.r.p.control_inwait.begin_o3()
        finally:
            self.release(row)
        self.finish()
        self.drain_remaining()

    def test_ready_census_is_not_an_applied_output_certificate(self):
        # Adversarial hook scheduling: real replies may arrive before Core is
        # allowed to apply its old output. No fake Future.done or worker row.
        self.assertIsNone(eh._idle_tick(self.r.core))
        eh.before_step_with_batch_queue(self.r.core)
        self.wire.sample_gate.set()
        for _ in range(200):
            eh._idle_tick(self.r.core)
            if self.r.p.phase == 'control_o3_cut':
                break
            time.sleep(.001)
        self.assertEqual(self.r.p.phase, 'control_o3_cut')
        self.assertEqual(len(self.r.p.control_inwait.cut), 1)
        self.assertFalse(self.applied)
        self.assertIsNone(self.r.d.admission_intent())
        self.finish()
        self.assertEqual(self.r.p.phase, 'native')
        self.drain_remaining()

    def test_fallback_hook_cannot_switch_before_later_outputs_apply(self):
        self.on_update = lambda: self.r.s.running.clear()
        self.release(self.probe())
        for _ in range(200):
            eh._idle_tick(self.r.core)
            if self.r.p.phase == 'control_o3_rollback':
                break
            time.sleep(.001)
        self.assertEqual(self.r.p.phase, 'control_o3_rollback')
        self.assertTrue(self.r.core.batch_queue)
        eh._idle_tick(self.r.core)
        self.assertEqual(self.r.p.phase, 'control_o3_rollback')
        self.assertIsNone(self.r.d._control)
        self.assertTrue(self.r.d._split_ready)
        self.finish()
        self.assertEqual(self.r.p.phase, 'idle')


class CutOwnership(unittest.TestCase):
    def helper(self, update=lambda output, value: {}):
        scheduler = NS(update_from_output=update)
        policy = NS(scheduler=scheduler)
        helper = auto.ControlInWait(policy)
        helper.enable_o3()
        helper._start_count('split-kept-entry')
        helper.row['cut_frames_applied'] = 0
        return helper, scheduler

    def test_only_the_exact_output_marks_one_cut_entry_after_success(self):
        helper, scheduler = self.helper()
        expected, equal_value = NS(n=1), NS(n=1)
        helper.cut = [expected]
        scheduler.update_from_output(equal_value, None)
        self.assertEqual(helper.cut, [expected])
        scheduler.update_from_output(expected, None)
        scheduler.update_from_output(expected, None)
        self.assertEqual(helper.cut, [])
        self.assertEqual(helper.row['cut_frames_applied'], 1)

    def test_failed_update_retains_the_cut_and_duplicate_hook_is_refused(self):
        def fail(output, value):
            raise ValueError('fixture update failed')
        helper, scheduler = self.helper(fail)
        output = NS()
        helper.cut = [output]
        with self.assertRaisesRegex(ValueError, 'update failed'):
            scheduler.update_from_output(output, None)
        self.assertEqual(helper.cut, [output])
        with self.assertRaisesRegex(Exception, 'one owned scheduler output hook'):
            helper.enable_o3()

    def test_invalid_or_duplicate_cut_is_refused_before_it_is_owned(self):
        for queue in ([(None, object(), None)] * 3, ['not-a-stock-frame'],
                      [(None, None, None), (None, None, None)]):
            with self.subTest(queue_len=len(queue)):
                r = kept_rig()
                r.s.update_from_output = lambda output, value: {}
                helper = r.p.control_inwait
                helper.enable_o3()
                r.core.batch_queue = queue
                with self.assertRaisesRegex(Exception, 'stock-output-cut-invalid'):
                    helper.begin_o3()
                self.assertFalse(helper.o3_active)
                self.assertFalse(helper.cut)
                self.assertIsNone(helper.row)


class YieldCore(unittest.TestCase):
    setUp = CoreOrder.setUp
    finish = CoreOrder.finish
    probe = CoreOrder.probe
    release = CoreOrder.release
    drain_remaining = CoreOrder.drain_remaining
    flags = (False, False, False)
    add_waiter = False

    def make_rig(self):
        r = lo3.yield_rig()
        ts.mirror_stock_pause(r)
        r.c.append_yield = ay.sh.AppendYieldConfig(6, 1, 8, 4, enabled=True)
        r.c.append_no_block_stop, r.c.yield_no_append_owner, r.c.yield_window = self.flags
        append = ay.run_to_yield(r)
        if self.add_waiter:
            r.s.max_num_running_reqs = 8
            r.add_held(ay.OTHER_UID, 6)
        ay.to_append(r)
        ay.run_append(r)
        r.tick(lambda: r.p.phase == 'tail')
        lo3.split_handback_step(r, append)
        ts.landed(r, append)
        lo3.first_decode_step(r, append)
        r.core.batch_queue = deque()
        r.s.pause_state = ci.Pause.UNPAUSED
        r.p.old_pause = ci.Pause.UNPAUSED
        r.p.control_inwait = auto.ControlInWait(r.p)
        return r

    def test_yield_resumes_the_bulk_at_its_own_committed_frontier(self):
        frontier = self.r.bulk.num_computed_tokens
        generation = self.r.d.route_generation
        self.release(self.probe())
        self.finish()
        self.assertEqual(self.r.p.phase, 'native')
        self.assertEqual(self.r.d.admission_intent()['request_uid'], self.r.bulk.request_id)
        self.assertEqual(self.r.bulk.num_computed_tokens, frontier)
        self.assertEqual(frontier, 4)
        self.assertEqual(self.r.d.route_generation, generation)
        self.assertEqual(self.r.d.append_yield_phase(), 'resume')
        stats = self.r.p.control_inwait.rows['split-kept-yield']
        self.assertEqual((stats['armed'], stats['arms_with_later_frames']), (1, 1))
        self.assertEqual(self.r.p.control_inwait.cut, [])
        self.assertEqual(self.r.p.control_inwait.members, ())
        self.drain_remaining()

    def test_revoked_yield_falls_back_without_arming_or_moving_the_bulk_frontier(self):
        frontier = self.r.bulk.num_computed_tokens
        self.on_update = lambda: self.r.d.append_yield_end('fixture-revoked')
        self.release(self.probe())
        self.finish()
        self.assertEqual(self.r.p.phase, 'idle')
        self.assertIsNone(self.r.d.admission_intent())
        self.assertEqual(self.r.bulk.num_computed_tokens, frontier)
        self.assertFalse(self.r.d._split_ready)


class WindowCore(YieldCore):
    flags = (True, True, True)
    add_waiter = True

    def test_yield_resumes_the_bulk_at_its_own_committed_frontier(self):
        # Override the singleton expectation: Y2 legitimately takes its next append.
        frontier = self.r.bulk.num_computed_tokens
        self.release(self.probe())
        self.finish()
        self.assertEqual(self.r.p.phase, 'native')
        self.assertEqual(self.r.d.append_yield_phase(), 'append')
        # The cycle remains bound to its bulk; Y2's typed window selects the append.
        self.assertEqual(self.r.d.admission_intent()['request_uid'], self.r.bulk.request_id)
        self.assertEqual(self.r.d._yield['append'].request_id, ay.OTHER_UID)
        self.assertFalse(self.r.c._native_admission_disabled(self.r.s))
        transaction = self.r.admit()
        self.assertEqual(transaction.request_uid, ay.OTHER_UID)
        request = self.r.c.cancel(transaction, never_submitted=True)
        self.r.s.waiting.prepend_request(request)
        self.assertFalse(self.r.c.active)
        self.assertEqual(self.r.bulk.num_computed_tokens, frontier)
        self.assertTrue(self.r.core.batch_queue)
        self.drain_remaining()


class StagedCore(CoreOrder):
    def make_rig(self):
        with mock.patch.object(ay.Rig, 'DRIVER', 'staged'):
            return kept_rig()


class StagedYieldCore(YieldCore):
    def make_rig(self):
        with mock.patch.object(ay.Rig, 'DRIVER', 'staged'):
            return super().make_rig()


class StagedWindowCore(WindowCore):
    def make_rig(self):
        with mock.patch.object(ay.Rig, 'DRIVER', 'staged'):
            return super().make_rig()


class Composition(unittest.TestCase):
    def test_all_eight_a_y1_y2_combinations_on_the_real_kept_yield_path(self):
        for owner in (YieldCore, StagedYieldCore):
            for flags in itertools.product((False, True), repeat=3):
                with self.subTest(owner=owner.__name__, flags=flags):
                    case = owner('test_yield_resumes_the_bulk_at_its_own_committed_frontier')
                    case.flags = flags
                    result = unittest.TestResult()
                    case.run(result)
                    self.assertTrue(result.wasSuccessful(), str(result.errors + result.failures))


class Mutants(unittest.TestCase):
    def killed(self, owner, name, old, new, test):
        method = getattr(owner, name)
        source = textwrap.dedent(inspect.getsource(method))
        self.assertEqual(source.count(old), 1)
        scope = dict(method.__globals__)
        exec(compile(source.replace(old, new), '<control-o3-mutant>', 'exec'), scope)
        with mock.patch.object(owner, name, scope[name]):
            result = unittest.TestResult()
            CoreOrder(test).run(result)
        self.assertFalse(result.wasSuccessful(), 'mutant survived: ' + old)

    def test_revert_to_queue_drain(self):
        self.killed(auto.AutoArmPolicy, 'tick',
                    'self.control_inwait is not None and self.control_inwait.o3:', 'False:',
                    'test_kept_census_arms_with_later_decode_still_queued_without_route_change')

    def test_mark_applied_before_real_update_returns(self):
        source = textwrap.dedent(inspect.getsource(auto.ControlInWait.enable_o3))
        mutant = source.replace('        result = original(scheduler_output, model_runner_output)\n', '')
        mutant = mutant.replace('        return result', '        return original(scheduler_output, model_runner_output)')
        self.killed(auto.ControlInWait, 'enable_o3',
                    source, mutant,
                    'test_finite_cut_waits_for_update_to_return_not_just_future_or_queue_pop')

    def test_skip_captured_output_application(self):
        self.killed(auto.ControlInWait, 'poll_o3', 'if self.cut:', 'if False:',
                    'test_ready_census_is_not_an_applied_output_certificate')

    def test_skip_fallback_drain(self):
        self.killed(auto.ControlInWait, 'poll_o3', 'if p.engine.batch_queue:', 'if False:',
                    'test_fallback_hook_cannot_switch_before_later_outputs_apply')

    def test_skip_handback_output_proof(self):
        self.killed(auto.ControlInWait, 'poll_o3',
                    'if joined is None or joined.num_output_tokens <= 0:', 'if False:',
                    'test_missing_handback_output_or_revoked_ledger_never_arms')

    def test_skip_ledger_recheck(self):
        self.killed(auto.ControlInWait, 'poll_o3',
                    'if not set(p.cohort) <= allowed:', 'if False:',
                    'test_revoked_ledger_is_rechecked_after_captured_outputs_apply')


class Settings(unittest.TestCase):
    def test_orphan_or_invalid_switch_refuses_before_engine_access(self):
        for env in ({'FLASH_S5_CONTROL_O3_INWAIT': '1'},
                    {'FLASH_S5_CONTROL_O3_INWAIT': 'x'},
                    {'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_CONTROL_INWAIT': '1',
                     'FLASH_S5_CONTROL_O3_INWAIT': '1'}):
            with self.subTest(env=env), self.assertRaises(s5_startup.StartupError):
                s5_startup.install_engine(None, env=env)

    def test_off_installs_no_output_hook(self):
        original = lambda output, value: {}
        scheduler = NS(update_from_output=original)
        engine = NS(scheduler=scheduler, _flash_s5_startup_ready=True)
        policy = auto.install(engine, NS(controller=None), {auto.ENV: '1', 'FLASH_S5_CONTROL_INWAIT': '1'})
        self.assertIs(scheduler.update_from_output, original)
        self.assertFalse(hasattr(scheduler, '_flash_s5_control_cut'))
        self.assertFalse(policy.control_inwait.o3)

    def test_explicit_switch_installs_one_owned_output_hook(self):
        original = lambda output, value: {}
        scheduler = NS(update_from_output=original)
        engine = NS(scheduler=scheduler, _flash_s5_startup_ready=True)
        policy = auto.install(engine, NS(controller=None),
                              {auto.ENV: '1', 'FLASH_S5_CONTROL_INWAIT': '1', 'FLASH_S5_O3': '1',
                               'FLASH_S5_CONTROL_O3_INWAIT': '1'})
        self.assertTrue(policy.control_inwait.o3)
        self.assertIsNot(scheduler.update_from_output, original)
        self.assertIs(scheduler._flash_s5_control_cut, policy.control_inwait)

    def test_off_kept_split_trace_matches_genuine_a15_parent(self):
        path = os.environ.get('CONTROL_O3_PARENT_SOURCE')
        source = (Path(path).read_bytes() if path else subprocess.check_output(
            ['git', 'show', 'a15bed1d:patches/s5_engine/native_auto_arm.py'], cwd=ci.ROOT))
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         '8090de4a9c349c89666916358b0c26f61c3ec0e8a6366545a4d990b4e89f5ee5')
        parent = types.ModuleType('control_o3_frozen_parent')
        exec(compile(source, '<frozen-a15-parent>', 'exec'), parent.__dict__)
        traces = []
        for module in (parent, auto):
            with ay.auto_arm_workers():
                r = kept_rig()
                # Same real terminal/handback fixture state, genuine parent methods.
                r.p.__class__ = module.AutoArmPolicy
                r.p.control_inwait = module.ControlInWait(r.p)
                r.core.batch_queue = [(None, object(), None)]
                before = len(r.stack.sender.envelopes)
                r.p.tick()
                self.assertEqual(r.p.phase, 'o3_drain')
                self.assertEqual(len(r.stack.sender.envelopes), before)
                r.core.batch_queue.clear()  # The frozen output-drain leaf, only in this OFF comparison.
                r.tick(lambda: r.p.phase == 'native')
                trace = ci.commands(r)
                self.assertEqual(trace[-1], ('flash_s5_native_overlap', 'census'))
                traces.append((trace, r.p.snapshot(), list(r.core.pause_states)))
                self.assertFalse(hasattr(r.s, '_flash_s5_control_cut'))
        self.assertEqual(traces[0], traces[1])


if __name__ == '__main__':
    unittest.main()
