#!/usr/bin/env python3
"""stage1-step L1 and L2 on the real parts around them (discovery mode: enough to know they work, not a clean battery).

* ``LazyClass``: FlashS5LazyIntermediateTensors over the stock AsyncIntermediateTensors source from the pinned gpu_worker.py: the receive runs once, at the first ``.tensors``.
* ``GeneratedRows``: the generated worker and scheduler rows, their place, exact revert, and that they compile.
* ``MergeRule``: scheduler_hooks.decode_merge_cap, and the scheduler's own cap rule and decode cadence in a small replay: 8 decoders merge into even steps and split 4 + 4 afterwards.
  This is a replay of the rule, not the real schedule(): the box's flash-sched stats lines and merge counters read the real one.
* ``DriverRules``: the mark and the sync on the real OverlapSegmentDriver, per selection and per policy phase.
* ``Router``: the real attach_native_router, session and client: native_stage_local carrying lazy_recv and decode_merge, and every refusal (the one new client field pair of this queue).

CPU only; nothing here is box evidence.
"""
import ast
import os
from pathlib import Path
import types
import unittest
from unittest import mock

import executor_hooks as eh
import scheduler_hooks as sh_hooks
import stage_overlap as so
from stage_net import ed, sh
from stage_overlap import OverlapSegmentDriver, StagedAdmissionController
import test_e3_stage_local_control as c0ctl
from test_e3_stage_local_control import attach, process_env

ROOT = Path(__file__).resolve().parents[1]
STEP_ENV = {'FLASH_S5_LAZY_RECV': '1', 'FLASH_S5_DECODE_MERGE': '1', 'FLASH_S5_STAGE_LOCAL': '1', 'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_STAGE_OVERLAP': '1',
            'FLASH_S5_STAGE_OVERLAP_DEFER_BIND': '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst'}


def stock_worker_text():
    return (ROOT / 'reference/vllm-src/v1/worker/gpu_worker.py').read_text()


class _Base:
    """vllm.sequence.IntermediateTensors's slice and items, so this suite also runs without the image's torch (the image holds the real one: the engine self-run imports it)."""

    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.tensors[key]
        return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def items(self):
        return self.tensors.items()


def lazy_class():
    try:
        from vllm.sequence import IntermediateTensors as base
    except ImportError:
        base = _Base
    tree = ast.parse(stock_worker_text())
    stock = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AsyncIntermediateTensors')
    source = ast.get_source_segment(stock_worker_text(), stock)
    space = dict(IntermediateTensors=base, Handle=object, Callable=object, torch=types.SimpleNamespace(Tensor=object))
    exec('from __future__ import annotations\n' + source + '\n' + eh._LAZY_RECV_CLASS, space)
    return space['FlashS5LazyIntermediateTensors'], space['AsyncIntermediateTensors']


class LazyClass(unittest.TestCase):
    def setUp(self):
        self.cls, self.stock = lazy_class()
        self.calls = []

    def receive(self, tensors=None):
        def run():
            self.calls.append('receive')
            handle = types.SimpleNamespace(wait=lambda: self.calls.append('wait'))
            return (dict(a=[1, 2, 3]) if tensors is None else tensors), [handle], [lambda: self.calls.append('post')]
        return run

    def test_the_receive_runs_once_at_the_first_tensors(self):
        got = self.cls(self.receive())
        self.assertEqual(self.calls, [])
        self.assertEqual(got.tensors, dict(a=[1, 2, 3]))
        self.assertEqual(self.calls, ['receive', 'wait', 'post'])
        self.assertEqual(got.tensors, dict(a=[1, 2, 3]))
        self.assertEqual(self.calls, ['receive', 'wait', 'post'])
        got.flash_s5_require_received()

    def test_items_and_a_key_receive(self):
        got = self.cls(self.receive())
        self.assertEqual(dict(got.items()), dict(a=[1, 2, 3]))
        self.assertEqual(self.calls.count('receive'), 1)
        self.assertEqual(self.cls(self.receive())['a'], [1, 2, 3])

    def test_a_slice_is_a_plain_received_view(self):
        got = self.cls(self.receive())[:2]
        self.assertIs(type(got), self.stock)
        self.assertEqual(got.tensors, dict(a=[1, 2]))
        self.assertEqual(self.calls, ['receive', 'wait', 'post'])

    def test_a_runner_that_never_read_it_fails_closed(self):
        got = self.cls(self.receive())
        with self.assertRaisesRegex(RuntimeError, 'never consumed'):
            got.flash_s5_require_received()
        self.assertEqual(self.calls, [])

    def test_a_receive_with_no_tensors_fails(self):
        def bad():
            return None, [], []
        with self.assertRaises(AssertionError):
            self.cls(bad).tensors

    def test_the_mark(self):
        for mark, want in ((True, True), (False, False), (1, False), ('1', False)):
            self.assertIs(eh.lazy_recv_marked(types.SimpleNamespace(flash_s5_lazy_recv=mark)), want)
        self.assertIs(eh.lazy_recv_marked(types.SimpleNamespace()), False)


class GeneratedRows(unittest.TestCase):
    def test_the_worker_rows(self):
        text = stock_worker_text()
        out = eh.generate_worker(text)
        self.assertEqual(eh.revert_worker(out), text)
        compile(out, 'gpu_worker.py', 'exec')
        self.assertLess(out.index('class FlashS5LazyIntermediateTensors'), out.index('class Worker(WorkerBase)'))
        marked = out.index('_flash_s5_hooks.lazy_recv_marked(scheduler_output)')
        lazy = out.index('intermediate_tensors = FlashS5LazyIntermediateTensors(', marked)
        stock = out.index('elif forward_pass and not get_pp_group().is_first_rank:', lazy)
        runner = out.index('scheduler_output, intermediate_tensors\n', stock)
        guard = out.index('intermediate_tensors.flash_s5_require_received()', runner)
        self.assertLess(guard, out.index('self.use_v2_model_runner', guard))

    def test_the_scheduler_row(self):
        import mirror_hooks
        import telemetry_hooks
        source = (ROOT / 'reference/vllm-src' / sh.SCHEDULER_PATH).read_text()
        e2 = mirror_hooks.scheduler(telemetry_hooks.scheduler(source))
        out = sh.generate_scheduler(e2)
        self.assertEqual(sh.revert_scheduler(out), e2)
        compile(out, 'scheduler.py', 'exec')
        cap = out.index('dec_cap = (ndec_all + 1) // 2 if ndec_all > 1 else None')
        row = out.index('dec_cap = _flash_s5_hooks.decode_merge_cap(self, dec_cap)')
        self.assertLess(cap, row)
        self.assertLess(row, out.index('ndec_sched = 0'))
        self.assertIn('decode_merge_cap', sh_hooks.__all__)


class Decoder:
    def __init__(self, eligible):
        self.next_decode_eligible_step = eligible
        self.is_prefill_chunk = False


def replay_step(scheduler, running, pp_size=2):
    """The scheduler's decode rule for one step (scheduler.py: the parity cap, the hook, the eligibility and cap skips, the cadence): who runs. A replay, not schedule()."""
    ndec_all = len(running)
    dec_cap = (ndec_all + 1) // 2 if ndec_all > 1 else None
    dec_cap = sh_hooks.decode_merge_cap(scheduler, dec_cap)
    ran = 0
    for request in running:
        if scheduler.current_step < request.next_decode_eligible_step:
            continue
        if dec_cap is not None and ran >= dec_cap:
            continue
        request.next_decode_eligible_step = scheduler.current_step + pp_size
        ran += 1
    return ran


class MergeRule(unittest.TestCase):
    def test_the_cap(self):
        scheduler = types.SimpleNamespace(current_step=4)
        self.assertEqual(sh_hooks.decode_merge_cap(scheduler, 4), 4)                 # unset: stock
        self.assertIsNone(sh_hooks.decode_merge_cap(scheduler, None))
        scheduler.flash_s5_decode_merge = False
        self.assertEqual(sh_hooks.decode_merge_cap(scheduler, 4), 4)
        scheduler.flash_s5_decode_merge = True
        self.assertIsNone(sh_hooks.decode_merge_cap(scheduler, 4))
        scheduler.current_step = 5
        self.assertEqual(sh_hooks.decode_merge_cap(scheduler, 4), 0)

    def test_eight_decoders_merge_and_split_again(self):
        running = [Decoder(0) for _ in range(8)]
        scheduler = types.SimpleNamespace(current_step=0)
        trace = []

        def steps(count):
            for _ in range(count):
                trace.append(replay_step(scheduler, running))
                scheduler.current_step += 1
        steps(8)
        self.assertEqual(trace[-4:], [4, 4, 4, 4])
        scheduler.flash_s5_decode_merge = True
        del trace[:]
        steps(8)
        self.assertEqual(trace[:3], [4, 0, 8])                                        # the odd cohort ran on the step before: it waits one extra step, once
        self.assertEqual(trace[3:], [0, 8, 0, 8, 0])
        scheduler.flash_s5_decode_merge = False
        del trace[:]
        steps(8)
        self.assertEqual(trace[:2], [4, 4])                                           # two steps after the end of the native phase: 4 + 4
        self.assertEqual(trace[2:], [4, 4, 4, 4, 4, 4])

    def test_merging_from_an_odd_step_loses_no_decoder(self):
        running = [Decoder(0) for _ in range(8)]
        scheduler = types.SimpleNamespace(current_step=1, flash_s5_decode_merge=True)
        self.assertEqual(replay_step(scheduler, running), 0)
        scheduler.current_step = 2
        self.assertEqual(replay_step(scheduler, running), 8)


class DriverBase(c0ctl.Base):
    async def asyncSetUp(self):
        with mock.patch.object(ed, 'NativeSegmentDriver', OverlapSegmentDriver), \
                mock.patch.object(sh, 'PrefillAdmissionController', StagedAdmissionController), \
                mock.patch.object(c0ctl, 'StagedSegmentDriver', OverlapSegmentDriver), \
                mock.patch.dict(os.environ, STEP_ENV):
            await super().asyncSetUp()


class DriverRules(DriverBase):
    async def test_off_until_selected(self):
        driver = self.driver
        self.assertEqual((driver.lazy_recv_installed, driver.decode_merge_installed, driver.lazy_recv, driver.decode_merge), (True, True, False, 'off'))
        output = types.SimpleNamespace(total_num_scheduled_tokens=8)
        driver.mark_lazy_recv(output)
        self.assertFalse(hasattr(output, 'flash_s5_lazy_recv'))
        scheduler = types.SimpleNamespace()
        driver.sync_decode_merge(scheduler)
        self.assertFalse(hasattr(scheduler, 'flash_s5_decode_merge'))              # off and never on: the attribute stays absent, the hook is stock
        self.assertEqual(driver.decode_step_counters, dict(lazy_marked=0, merge_steps=0, merge_on=0, merge_off=0))

    async def test_the_mark_skips_an_empty_step(self):
        driver = self.driver
        driver.lazy_recv = True
        full, empty = types.SimpleNamespace(total_num_scheduled_tokens=8), types.SimpleNamespace(total_num_scheduled_tokens=0)
        driver.mark_lazy_recv(full)
        driver.mark_lazy_recv(empty)
        self.assertIs(full.flash_s5_lazy_recv, True)
        self.assertFalse(hasattr(empty, 'flash_s5_lazy_recv'))
        self.assertEqual(driver.decode_step_counters['lazy_marked'], 1)

    async def test_chunk_follows_the_policy_phase_and_always_does_not(self):
        driver = self.driver
        policy = types.SimpleNamespace(phase='idle', manual_owned=False)
        scheduler = types.SimpleNamespace()
        with mock.patch.object(driver, 'auto_arm', policy, create=True):
            driver.decode_merge = 'chunk'
            driver.sync_decode_merge(scheduler)
            self.assertFalse(getattr(scheduler, 'flash_s5_decode_merge', False))
            policy.phase = 'native'
            driver.sync_decode_merge(scheduler)
            self.assertIs(scheduler.flash_s5_decode_merge, True)
            driver.sync_decode_merge(scheduler)
            policy.phase = 'tail'
            driver.sync_decode_merge(scheduler)
            self.assertIs(scheduler.flash_s5_decode_merge, False)
            self.assertEqual(driver.decode_step_counters, dict(lazy_marked=0, merge_steps=2, merge_on=1, merge_off=1))
            policy.phase, policy.manual_owned = 'native', True
            driver.sync_decode_merge(scheduler)
            self.assertIs(scheduler.flash_s5_decode_merge, False)                 # a manual prefill is not the policy's
            policy.phase, policy.manual_owned = 'idle', False
            driver.decode_merge = 'always'
            driver.sync_decode_merge(scheduler)
            self.assertIs(scheduler.flash_s5_decode_merge, True)

    async def test_a_switch_needs_the_deferred_bind_at_startup(self):
        for name in ('FLASH_S5_LAZY_RECV', 'FLASH_S5_DECODE_MERGE'):
            check = so.lazy_recv_enabled if name == 'FLASH_S5_LAZY_RECV' else so.decode_merge_enabled
            self.assertIs(check({}), False)
            self.assertIs(check({name: '0'}), False)
            self.assertIs(check({name: '1', 'FLASH_S5_STAGE_OVERLAP': '1', 'FLASH_S5_STAGE_LOCAL': '1', 'FLASH_S5_STAGE_OVERLAP_DEFER_BIND': '1'}), True)
            with self.assertRaises(so.DriverError):
                check({name: '2'})
            with self.assertRaises(so.DriverError):
                check({name: '1'})


class Router(DriverBase):
    """The box client's new fields through the real router (A1-h's lesson)."""

    ALL = dict(c0ctl.BOTH, **STEP_ENV)
    C1 = dict(local=True, overlap=True, defer_bind=True)

    async def test_native_stage_local_carries_both_fields(self):
        with process_env(self.ALL), mock.patch.dict('os.environ', STEP_ENV):
            app, endpoint = attach(self.frontend, '1', '1')
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            for lazy, merge in ((False, 'off'), (True, 'off'), (True, 'chunk'), (True, 'always')):
                code, reply = await self.post(endpoint, app, 'native_stage_local', lazy_recv=lazy, decode_merge=merge, **self.C1)
                self.assertEqual(code, 200, reply)
                self.assertEqual((reply['lazy_recv'], reply['decode_merge'], reply['defer_bind']), (lazy, merge, True))
                self.assertEqual((self.driver.lazy_recv, self.driver.decode_merge), (lazy, merge))
                code, status = await self.post(endpoint, app, 'status')
                self.assertEqual(code, 200)
                block = status['native_stage_local']['overlap']['decode_step']
                self.assertEqual((block['lazy_recv_installed'], block['lazy_recv'], block['decode_merge_installed'], block['decode_merge']), (True, lazy, True, merge))
                self.assertEqual(sorted(block['counters']), ['lazy_marked', 'merge_off', 'merge_on', 'merge_steps'])
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)

    async def test_bad_values_and_a_selection_without_the_deferred_bind_are_refused_not_dropped(self):
        with process_env(self.ALL), mock.patch.dict('os.environ', STEP_ENV):
            app, endpoint = attach(self.frontend, '1', '1')
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            for bad in (dict(self.C1, lazy_recv=1), dict(self.C1, decode_merge='on'), dict(self.C1, decode_merge=True), dict(local=True, overlap=True, lazy_recv=True),
                        dict(local=True, overlap=True, decode_merge='chunk')):
                code, reply = await self.post(endpoint, app, 'native_stage_local', **bad)
                self.assertNotEqual(code, 200, (bad, reply))
            self.assertEqual((self.driver.lazy_recv, self.driver.decode_merge), (False, 'off'))
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)

    async def test_not_installed_refuses_each_field(self):
        for name, field, value in (('FLASH_S5_LAZY_RECV', 'lazy_recv', True), ('FLASH_S5_DECODE_MERGE', 'decode_merge', 'chunk')):
            env = {**STEP_ENV, name: '0'}
            with process_env(dict(c0ctl.BOTH, **env)), mock.patch.dict('os.environ', env):
                app, endpoint = attach(self.frontend, '1', '1')
                self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
                code, reply = await self.post(endpoint, app, 'native_stage_local', **{field: value}, **self.C1)
                self.assertNotEqual(code, 200, reply)
                self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)


del DriverBase

if __name__ == '__main__':
    unittest.main()
