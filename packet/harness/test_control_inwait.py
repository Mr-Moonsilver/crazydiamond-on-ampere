"""CPU ordering proof for FULL observation overlap, not CUDA completion evidence.

Uses the actual auto policy, admission controller, segment driver and four bridge
workers. Reused GPU/model/graph leaves are declared in control_inwait_fixture_leaves.json.
The connected Core tests below execute the pinned, generated queue/result/update body.
"""
import ast
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
import enum
import hashlib
import inspect
import itertools
import os
from pathlib import Path
from queue import Queue
import subprocess
import threading
import textwrap
import time
import types
import unittest
from unittest import mock
from typing import cast

import test_append_yield as ay
import test_e3_executor_threads as rpc
import executor_hooks as eh
import native_auto_arm as auto
import s5_startup

ROOT = Path(__file__).resolve().parents[1]
NS = types.SimpleNamespace


def frozen_parent():
    path = os.environ.get('CONTROL_INWAIT_PARENT_SOURCE')
    source = (Path(path).read_bytes() if path else subprocess.check_output(
        ['git', 'show', '46d28946:patches/s5_engine/native_auto_arm.py'], cwd=ROOT))
    assert hashlib.sha256(source).hexdigest() == '3d5c813856f0c7acd68225a2df69c157d7d2aca2b653ea19ee0586ea4e1f9672'
    parent = types.ModuleType('control_inwait_parent_auto')
    exec(compile(source, '<frozen-parent-native_auto_arm>', 'exec'), parent.__dict__)
    return parent


class Pause(str, enum.Enum):
    UNPAUSED = 'UNPAUSED'
    PAUSED_NEW = 'PAUSED_NEW'
    PAUSED_ALL = 'PAUSED_ALL'


def full_rig(module=auto, enabled=True):
    r = ay.Rig()
    # Enter the starting FULL bank through the real all-rank transition.
    r.p._internal = True
    r.core.paused = True
    r.d._disarm_admission('fixture-full-start')
    r.control({'action': 'leave_split'})
    r.p = module.AutoArmPolicy(r.core, r.d)
    r.d.auto_arm = r.p
    if enabled:
        r.p.control_inwait = auto.ControlInWait(r.p)
    r.s.pause_state = Pause.UNPAUSED
    r.core.paused = False
    old = r.s.set_pause_state

    def pause(state):
        old(state)
        r.s.pause_state = state
    r.s.set_pause_state = pause
    r.stock_handback(ay.Req('decoder', 8))
    r.stack.sender.envelopes.clear()
    return r


def commands(r):
    return [(name, args[0].get('action') if isinstance(args[0], dict) else None)
            for name, args, _, _ in r.stack.sender.envelopes]


class Observation(unittest.TestCase):
    def setUp(self):
        context = mock.patch.object(auto, 'MIN_NATIVE_ROWS', 1)
        context.start(); self.addCleanup(context.stop)
        context = ay.auto_arm_workers()
        context.__enter__(); self.addCleanup(context.__exit__, None, None, None)
        self.r = full_rig()

    def tick(self):
        r = self.r
        result = eh._idle_tick(r.core)
        r.d.advance()
        r.stack.complete_everything()
        return result

    def certify(self):
        for _ in range(20):
            self.tick()
            if self.r.p.control_inwait.certified:
                return
        self.fail('identity never certified')

    def arm(self):
        for _ in range(20):
            self.tick()
            if self.r.p.phase == 'native':
                return
        self.fail('entry never armed')

    def test_identity_submits_before_queue_empty_and_does_not_bypass_decode(self):
        r = self.r
        r.core.batch_queue = [object()]
        self.assertIsNone(self.tick())
        self.assertEqual(r.p.phase, 'control_observe')
        self.assertEqual(r.s.pause_state, Pause.PAUSED_NEW)
        self.assertEqual(commands(r), [('flash_s5_native_overlap', 'identity')])
        self.certify()
        self.assertEqual(r.s.pause_state, Pause.PAUSED_ALL)
        for _ in range(3):
            self.tick()
        self.assertEqual(commands(r), [('flash_s5_native_overlap', 'identity')])
        self.assertIsNone(r.d.admission_intent())
        r.core.batch_queue.clear()
        self.arm()
        self.assertNotIn(('flash_s5_native_overlap', 'barrier'), commands(r))
        self.assertIn(('flash_s5_transition', None), commands(r))
        self.assertEqual(len(r.d._last_transition_acks), 4)
        self.assertTrue(all(ack.drained_streams > 0 for ack in r.d._last_transition_acks.values()))
        row = r.p.snapshot()['control_inwait']['transitions']['full-entry']
        self.assertEqual((row['armed'], row['drains_skipped']), (1, 1))
        self.assertEqual(row['drains_kept'], {'applied-output-cut': 1, 'typed-route-fence': 1})
        self.assertGreaterEqual(row['pause_to_arm_ms'], row['decoder_stall_ms'])

    def test_cohort_finish_after_query_before_output_application_cannot_arm(self):
        r = self.r
        r.core.batch_queue = [object()]
        self.certify()
        r.s.running.clear()  # The queued sample's actual scheduler update.
        r.core.batch_queue.clear()
        self.tick()
        self.assertNotEqual(r.p.phase, 'native')
        self.assertIsNone(r.d.admission_intent())
        self.assertFalse(r.d._split_ready)
        self.assertNotIn(('flash_s5_transition', None), commands(r))

    def test_same_uid_replacement_and_generation_invalidate_the_certificate(self):
        for mutation in ('request', 'generation'):
            with self.subTest(mutation=mutation):
                self.r = r = full_rig()
                self.certify()
                if mutation == 'request':
                    r.s.running[:] = [ay.Req('decoder', 8, computed=8)]
                else:
                    r.d.route_generation += 1
                self.assertFalse(r.p.control_inwait.current())
                # A fresh probe, never enter_split using the old observation.
                self.tick()
                self.assertEqual(r.p.phase, 'control_observe')
                self.assertEqual(commands(r), [('flash_s5_native_overlap', 'identity')] * 2)

    def test_public_manual_control_cannot_borrow_the_overlap_permission(self):
        self.tick()
        with self.assertRaisesRegex(Exception, 'cycle-owned'):
            self.r.d.submit_control({'action': 'identity'})

    def test_worker_bad_census_and_bad_capability_never_arm(self):
        for mutate in ('prefill', 'counts', 'uid'):
            with self.subTest(mutate=mutate):
                self.r = r = full_rig()
                if mutate == 'prefill':
                    r.stack.workers[1].model_runner.req_states.num_computed_prefill_tokens[0] = 0
                    self.tick()
                elif mutate == 'counts':
                    with ay.auto_arm_workers(counts=()):
                        self.tick()
                else:
                    state = r.stack.workers[1].model_runner.req_states
                    state.req_id_to_index['foreign-decoder'] = state.req_id_to_index.pop('decoder')
                    self.tick()
                for _ in range(4):
                    self.tick()
                self.assertIsNone(r.d.admission_intent())
                self.assertFalse(r.d._split_ready)
                reason = 'worker-graph-support-refused' if mutate == 'counts' else 'worker-barrier-cohort-refused'
                self.assertIn(reason, r.p.rollbacks)

    def test_timeout_retains_ownership_and_never_switches(self):
        self.tick()
        r = self.r
        r.p.deadline = r.d.clock() - 1
        with self.assertRaisesRegex(Exception, 'deadline-expired; ownership retained'):
            self.tick()
        self.assertNotIn(('flash_s5_transition', None), commands(r))
        self.assertEqual(r.p.phase, 'control_observe')

    def test_counter_snapshot_is_detached_and_bounded(self):
        self.arm()
        ci = self.r.p.control_inwait
        self.assertEqual(self.r.d._native_admission_record()['control_inwait'], ci.snapshot())
        snapshot = ci.snapshot()
        snapshot['transitions']['full-entry']['drains_kept'].clear()
        self.assertEqual(len(ci.rows['full-entry']['drains_kept']), 2)
        self.assertEqual(len(ci.rows), 1)
        self.assertEqual(ci.members, ())

    def test_missing_route_fence_is_not_replaced_by_the_identity(self):
        self.certify()
        self.r.stack.workers[2].dispatcher.native.controller.drain = lambda: 0
        with self.assertRaisesRegex(Exception, 'transition-fence-unproved; ownership retained'):
            self.arm()
        self.assertIsNone(self.r.d.admission_intent())
        self.assertEqual(self.r.p.phase, 'split')


class OffAndConfiguration(unittest.TestCase):
    def test_install_switch_is_strict_and_allocates_nothing_when_off(self):
        for setting in (None, '0', '1'):
            engine = NS(scheduler=object(), _flash_s5_startup_ready={'engine_epoch': 7})
            driver = NS(controller=object())
            env = {auto.ENV: '1'}
            if setting is not None:
                env['FLASH_S5_CONTROL_INWAIT'] = setting
            policy = auto.install(engine, driver, env)
            self.assertEqual(policy.control_inwait is not None, setting == '1')
            self.assertEqual('control_inwait' in policy.__dict__, setting == '1')
        for setting in ('true', '', 1, None):
            with self.subTest(setting=setting), self.assertRaisesRegex(Exception, 'invalid boolean'):
                auto.install(NS(scheduler=None, _flash_s5_startup_ready=True), NS(controller=None),
                             {auto.ENV: '1', 'FLASH_S5_CONTROL_INWAIT': setting})

    def test_startup_rejects_invalid_or_orphan_flag_before_engine_access(self):
        class Untouched:
            def __getattribute__(self, name):
                raise AssertionError(name)
        for value in ('true', '', 1, None, '1'):
            with self.subTest(value=value), self.assertRaises(s5_startup.StartupError):
                s5_startup.install_engine(Untouched(), env={'FLASH_S5_CONTROL_INWAIT': value})

    def test_disabled_command_pause_and_status_trace_matches_frozen_parent(self):
        parent = frozen_parent()
        traces = []
        for module in (parent, auto):
            with mock.patch.object(module, 'MIN_NATIVE_ROWS', 1), ay.auto_arm_workers():
                r = full_rig(module, enabled=False)
                for _ in range(20):
                    r.p.tick(); r.d.advance(); r.stack.complete_everything()
                    if r.p.phase == 'native':
                        break
                self.assertEqual(r.p.phase, 'native')
                self.assertNotIn('control_inwait', r.d._native_admission_record())
                traces.append((commands(r), list(r.core.pause_states), r.p.snapshot()))
        self.assertEqual(traces[0], traces[1])

    def test_frozen_parent_negative_identity_cannot_overlap_an_outstanding_sample(self):
        parent = frozen_parent()
        with mock.patch.object(parent, 'MIN_NATIVE_ROWS', 1), ay.auto_arm_workers():
            r = full_rig(parent, enabled=False)
            r.core.batch_queue = [object()]
            r.p.tick()
            self.assertEqual(r.p.phase, 'drain')
            self.assertEqual(commands(r), [])
            self.assertEqual(r.s.pause_state, Pause.PAUSED_ALL)


class YieldComposition(unittest.TestCase):
    def setUp(self):
        context = ay.auto_arm_workers()
        context.__enter__(); self.addCleanup(context.__exit__, None, None, None)

    def prepare(self, a=False, y1=False, y2=False):
        r = ay.Rig(max_new=6)  # Bulk rows left=7: a genuine bulk under either Y1 setting.
        r.c.append_no_block_stop = a
        r.c.yield_no_append_owner = y1
        r.c.yield_window = y2
        r.p.control_inwait = auto.ControlInWait(r.p)
        r.p.old_pause = Pause.UNPAUSED
        old = r.s.set_pause_state

        def pause(state):
            old(state)
            r.s.pause_state = state
        r.s.set_pause_state = pause
        r.s.pause_state = Pause.UNPAUSED
        append = ay.run_to_yield(r)
        ay.to_append(r)
        ay.run_append(r)
        ay.to_full(r)
        r.stock_handback(append)
        r.stack.sender.envelopes.clear()
        return r

    def test_all_eight_a_y1_y2_combinations_reenter_with_exact_bulk_and_census(self):
        for flags in itertools.product((False, True), repeat=3):
            with self.subTest(flags=flags):
                r = self.prepare(*flags)
                r.tick(lambda: r.p.phase == 'native' and r.d.admission_intent() is not None)
                self.assertEqual(r.d.admission_intent()['request_uid'], r.bulk.request_id)
                self.assertEqual(r.p.cohort, (ay.APPEND_UID,))
                self.assertNotIn(('flash_s5_native_overlap', 'barrier'), commands(r))
                row = r.p.snapshot()['control_inwait']['transitions']['yield-full-reentry']
                self.assertEqual(row['armed'], 1)
                self.assertEqual(row['drains_skipped'], 1)

    def test_yield_revoked_while_identity_pending_cannot_arm_a_bulk(self):
        r = self.prepare()
        r.tick(lambda: r.p.phase == 'control_observe')
        r.d.append_yield_end('fixture-client-left')
        r.tick(lambda: r.p.phase == 'idle')
        self.assertIsNone(r.d.admission_intent())
        self.assertFalse(r.d._split_ready)
        self.assertEqual(r.p.control_inwait.members, ())

    def test_failed_cpu_prefill_census_needs_a_real_decode_update_not_another_drain(self):
        r = self.prepare()
        for worker in r.stack.workers.values():
            state = worker.model_runner.req_states
            index = state.req_id_to_index[ay.APPEND_UID]
            state.num_computed_prefill_tokens[index] -= 1
        r.tick(lambda: r.p.phase == 'drain_wait')
        self.assertEqual(r.p.reentry_refusals, 1)
        self.assertIsNone(r.d.admission_intent())
        # Ordinary stock's first decode refreshes the CPU prefill census.
        for worker in r.stack.workers.values():
            state = worker.model_runner.req_states
            index = state.req_id_to_index[ay.APPEND_UID]
            state.num_computed_prefill_tokens[index] = state.prefill_len.np[index]
        r.c._schedule_seq += 1
        r.tick(lambda: r.p.phase == 'native' and r.d.admission_intent() is not None)
        self.assertEqual(commands(r).count(('flash_s5_native_overlap', 'identity')), 2)
        self.assertNotIn(('flash_s5_native_overlap', 'barrier'), commands(r))


def core_step():
    source = eh.generate_core((ROOT / 'reference/vllm-src' / eh.CORE_PATH).read_text())
    owner = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'EngineCore')
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == 'step_with_batch_queue')
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                            method], type_ignores=[])
    scope = dict(_flash_s5_hooks=eh, cast=cast, Future=Future)
    exec(compile(ast.fix_missing_locations(tree), eh.CORE_PATH, 'exec'), scope)
    return scope['step_with_batch_queue']


class OrderedWorkers(rpc.Broadcast):
    """In-memory MQ leaf: four ordered worker threads, real native RPC bodies."""

    def __init__(self, executor, rig):
        super().__init__(executor)
        self.rig = rig
        self.sample_gate = threading.Event()
        self.queues = [Queue() for _ in range(4)]
        self.order = [[] for _ in range(4)]
        self.errors = []
        self.threads = [threading.Thread(target=self.run, args=(rank,)) for rank in range(4)]
        for thread in self.threads:
            thread.start()

    def enqueue(self, item):
        self.writes.append(item)
        for queue in self.queues:
            queue.put(item)

    def run(self, rank):
        worker = self.rig.stack.workers[rank]
        runner = worker.model_runner
        while (item := self.queues[rank].get()) is not None:
            name, args, kwargs, reply_rank = item
            self.order[rank].append(name)
            try:
                if name == 'execute_model':
                    runner.execute_model_state = NS()
                    # Pinned ordinary update_requests refreshes the CPU prefill census.
                    for index in runner.req_states.req_id_to_index.values():
                        runner.req_states.num_computed_prefill_tokens[index] = runner.req_states.prefill_len.np[index]
                    value = 'executed'
                elif name == 'sample_tokens':
                    if not self.sample_gate.wait(5):
                        raise TimeoutError('fixture sample gate')
                    runner.execute_model_state = None
                    value = 'sampled'
                else:
                    value = getattr(worker, name)(*args, **kwargs)
                answer = (self.e.SUCCESS, value)
            except BaseException as error:
                self.errors.append(error)
                answer = ('fixture-error', repr(error))
            if reply_rank is None or reply_rank == rank:
                self.e.response_mqs[rank].q.put(answer)

    def close(self):
        self.sample_gate.set()
        for queue in self.queues:
            queue.put(None)
        for thread in self.threads:
            thread.join(6)
            if thread.is_alive():
                raise AssertionError('worker fixture did not exit')


class RealCoreOrder(unittest.TestCase):
    def make_rig(self):
        return full_rig()

    def setUp(self):
        context = mock.patch.object(auto, 'MIN_NATIVE_ROWS', 1)
        context.start(); self.addCleanup(context.stop)
        context = ay.auto_arm_workers()
        context.__enter__(); self.addCleanup(context.__exit__, None, None, None)
        self.r = r = self.make_rig()
        self.executor, self.transport = rpc.install()
        self.addCleanup(self.transport.close, 1)
        self.wire = OrderedWorkers(self.executor, r)
        self.addCleanup(self.wire.close)
        self.executor.rpc_broadcast_mq = self.wire
        self.executor._flash_s5_bounded_send = eh.BoundedSender(self.wire)
        self.executor.output_rank = 3
        self.executor.kv_output_aggregator = self.executor.ec_output_aggregator = None
        env = NS(VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3)
        for name in ('execute_model', 'sample_tokens'):
            method = rpc.method(eh.EXECUTOR_PATH, 'MultiprocExecutor', name, {'envs': env})
            setattr(self.executor, name, types.MethodType(method, self.executor))
        r.d.executor = self.executor
        self.applied = []
        self.on_update = lambda: None
        r.s.has_requests = lambda: r.s.pause_state != Pause.PAUSED_ALL and bool(r.s.running)
        r.s.schedule = lambda throttle: NS(total_num_scheduled_tokens=1, pending_structured_output_tokens=False)
        r.s.get_grammar_bitmask = lambda output: None

        def update(output, value):
            self.applied.append(value)
            self.on_update()
            return {}
        r.s.update_from_output = update
        r.core.batch_queue = deque()
        r.core.batch_queue_size = 2
        r.core.model_executor = self.executor
        r.core.is_ec_consumer = True
        r.core.is_pooling_model = False
        r.core.check_for_draft_tokens = False
        r.core._should_throttle_prefills = lambda: False
        r.core.log_error_detail = lambda output: nullcontext()
        r.core.capture_iteration_details = lambda output: nullcontext(None)
        r.core._attach_iteration_details = lambda *args: None
        r.core._process_aborts_queue = lambda: None
        self.step = core_step()
        # Fill one real stock slot before the next top-of-Core auto-arm tick.
        raw = rpc.method(eh.CORE_PATH, 'EngineCore', 'step_with_batch_queue', {'cast': cast, 'Future': Future})
        raw(r.core)
        self.assertEqual(len(r.core.batch_queue), 1)

    def overlap_probe(self):
        row = rpc.threaded(lambda: self.step(self.r.core))
        deadline = time.monotonic() + 2
        while not any(item[0] == 'flash_s5_native_overlap' for item in self.wire.writes):
            if time.monotonic() > deadline:
                self.wire.sample_gate.set()
                rpc.join(self, row)
                self.fail('identity not submitted behind outstanding sample')
            time.sleep(.001)
        self.assertFalse(self.applied)
        self.assertEqual([item[0] for item in self.wire.writes[:3]],
                         ['execute_model', 'sample_tokens', 'flash_s5_native_overlap'])
        self.assertIsNone(self.r.d.admission_intent())
        self.wire.sample_gate.set()
        _, errors = rpc.join(self, row)
        self.assertEqual(errors, [])

    def finish(self):
        for _ in range(200):
            self.step(self.r.core)
            if self.r.p.phase in ('native', 'idle'):
                return
            time.sleep(.001)
        self.fail('Core did not settle')

    def test_real_four_worker_fifo_query_overlaps_decode_and_switch_follows_applied_cut(self):
        self.overlap_probe()
        self.finish()
        self.assertEqual(self.r.p.phase, 'native')
        self.assertGreaterEqual(len(self.applied), 2)
        self.assertFalse(self.r.core.batch_queue)
        for order in self.wire.order:
            query = order.index('flash_s5_native_overlap')
            switch = order.index('flash_s5_transition')
            self.assertEqual(order[query - 2:query], ['execute_model', 'sample_tokens'])
            self.assertEqual(order[switch - 1], 'sample_tokens')
        self.assertFalse(self.wire.errors)
        self.assertEqual(self.transport._calls, {})
        self.assertEqual(self.executor._flash_s5_stock_futures, {})

    def test_sample_done_is_not_applied_cohort_proof(self):
        # The query is ordered after the sample, but only this real Core update
        # removes its finished decoder. The old worker census must never arm it.
        self.on_update = lambda: self.r.s.running.clear()
        self.overlap_probe()
        self.finish()
        self.assertEqual(self.r.p.phase, 'idle')
        self.assertFalse(self.r.d._split_ready)
        self.assertIsNone(self.r.d.admission_intent())
        self.assertNotIn('flash_s5_transition', [item[0] for item in self.wire.writes])
        self.assertFalse(self.wire.errors)

    def test_equal_count_uid_change_during_output_application_cannot_arm_old_cohort(self):
        self.on_update = lambda: self.r.s.running.__setitem__(slice(None), [ay.Req('new-decoder', 8, computed=8)])
        self.overlap_probe()
        self.finish()
        self.assertEqual(self.r.p.phase, 'idle')
        self.assertIsNone(self.r.d.admission_intent())
        self.assertNotIn('flash_s5_transition', [item[0] for item in self.wire.writes])

    def test_full_control_ring_refuses_before_registration_and_decode_is_not_bypassed(self):
        before = set(self.transport._calls)
        with mock.patch.object(self.wire.buffer, 'get_metadata',
                               lambda index: nullcontext(bytearray([1, 0, 0, 0, 0]))):
            self.assertIsNone(eh._idle_tick(self.r.core))
            self.assertIsNone(eh.before_step_with_batch_queue(self.r.core))
        self.assertGreater(self.executor._flash_s5_bounded_send.refused_full, 0)
        self.assertEqual(set(self.transport._calls), before)
        self.assertIsNone(self.r.d._control_seq)
        self.assertEqual([item[0] for item in self.wire.writes], ['execute_model', 'sample_tokens'])
        self.wire.sample_gate.set()
        self.finish()
        self.assertEqual(self.r.p.phase, 'native')
        self.assertEqual(self.transport._calls, {})

    def test_worker_query_error_propagates_and_never_enters_split(self):
        def refuse(_command):
            raise ValueError('fixture-worker-identity-failure')
        self.r.stack.workers[1].flash_s5_native_overlap = refuse
        self.wire.sample_gate.set()
        with self.assertRaises(Exception):
            self.finish()
        self.assertEqual(len(self.wire.errors), 1)
        self.assertIsNone(self.r.d.admission_intent())
        self.assertFalse(self.r.d._split_ready)
        self.assertNotIn('flash_s5_transition', [item[0] for item in self.wire.writes])

    def test_deferred_grammar_sample_is_applied_before_the_route_cut(self):
        self.r.s.schedule = lambda throttle: NS(total_num_scheduled_tokens=1, pending_structured_output_tokens=True)
        self.overlap_probe()
        self.finish()
        self.assertEqual(self.r.p.phase, 'native')
        self.assertGreaterEqual(len(self.applied), 2)
        self.assertFalse(self.r.core.batch_queue)
        self.assertFalse(self.wire.errors)


class Mutants(unittest.TestCase):
    def killed(self, owner, name, old, new, test):
        method = getattr(owner, name)
        source = textwrap.dedent(inspect.getsource(method))
        self.assertEqual(source.count(old), 1)
        scope = dict(method.__globals__)
        exec(compile(source.replace(old, new), '<control-inwait-mutant>', 'exec'), scope)
        with mock.patch.object(owner, name, scope[name]):
            result = unittest.TestResult()
            Observation(test).run(result)
        self.assertFalse(result.wasSuccessful(), 'mutant survived: ' + name + ' ' + old)

    def test_no_observation_overlap(self):
        self.killed(auto.AutoArmPolicy, 'tick',
                    "self.control_inwait if self.phase == 'drain' else None", 'None',
                    'test_identity_submits_before_queue_empty_and_does_not_bypass_decode')

    def test_route_before_applied_output_cut(self):
        self.killed(auto.AutoArmPolicy, 'tick',
                    '(observation is None or observation.certified)', '(observation is None)',
                    'test_identity_submits_before_queue_empty_and_does_not_bypass_decode')

    def test_stale_request_certificate(self):
        self.killed(auto.ControlInWait, 'current',
                    'and saved is r', '',
                    'test_same_uid_replacement_and_generation_invalidate_the_certificate')

    def test_stale_generation_certificate(self):
        self.killed(auto.ControlInWait, 'current',
                    'self.identity == (d.engine_epoch, d.route_generation)', 'True',
                    'test_same_uid_replacement_and_generation_invalidate_the_certificate')

    def test_skipping_cohort_capability_proof(self):
        self.killed(auto.ControlInWait, 'poll',
                    'if not p._proof(result, capability=True):', 'if False:',
                    'test_worker_bad_census_and_bad_capability_never_arm')

    def test_accepting_missing_device_fence(self):
        self.killed(auto.ControlInWait, 'acknowledge',
                    'if (not self.certified', 'if False and (not self.certified',
                    'test_missing_route_fence_is_not_replaced_by_the_identity')


if __name__ == '__main__':
    unittest.main()
