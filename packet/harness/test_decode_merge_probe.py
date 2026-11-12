#!/usr/bin/env python3
"""decode-merge probe (12 Nov 2026): cohort formation and liveness under decode_merge off / off-small, the liveness
observer, the arm guard and the ABBA table.

* ``Cohorts``: the real ``scheduler_hooks.decode_merge_cap`` and the real ``OverlapSegmentDriver.sync_decode_merge`` in
  a replay of the E2 scheduler's decode rule (patches/pdalt/make_pdalt.py:126-169 with FLASH_PARITY_BALANCE=1, the
  async PP cadence ``next_decode_eligible_step = step + pp_size``): who runs each step. A replay of the rule, not the
  real ``schedule()``; the box's ``flash-sched stats`` lines read the real one.
* ``Shrinking``: 8 -> 4 -> 2 -> 1 -> 0 live decoders, the drops on both step parities, by EOS (leaves running) and by
  cancel (leaves running mid-cohort), for every mode and around native-phase boundaries: no live decoder waits more
  than its cadence, none is lost, and the liveness observer stays quiet; a mutant cap that starves the odd cohort
  trips it.
* ``Observer``: DecodeLiveness counters, one trip and one line per episode, env parse, the status key only under the
  switch, and a chunk engine's step path unchanged.
* ``Control``: ``off-small`` through the real router; refusals unchanged.
* ``Guard``: decode_liveness_guard on fake polls and on the real compose4 replay loop with a stalled stream.
* ``Table``: n3dmp_packet over j2's configure.
CPU only; nothing here is box evidence.
"""
import contextlib
import io
import logging
import os
from pathlib import Path
import threading
import time
import types
import unittest
from unittest import mock

import scheduler_hooks as sh_hooks
import stage_overlap as so
from stage_overlap import DecodeLiveness, OverlapSegmentDriver
import test_stage1_step_engine as s1
import test_e3_stage_local_control as c0ctl
from test_e3_stage_local_control import attach, process_env
import decode_liveness_guard as guard_mod

HERE = Path(__file__).resolve().parent


class Decoder:
    _ids = 0

    def __init__(self, eligible=0):
        Decoder._ids += 1
        self.request_id = 'r%d' % Decoder._ids
        self.next_decode_eligible_step = eligible
        self.is_prefill_chunk = False
        self.num_output_tokens = 1


def replay_step(scheduler, running, pp_size=2):
    """make_pdalt.py:126-169 with parity balance on: the parity cap, the hook, the eligibility skip, the cap skip, cadence."""
    ndec_all = sum(1 for r in running if not r.is_prefill_chunk)
    dec_cap = (ndec_all + 1) // 2 if ndec_all > 1 else None
    dec_cap = sh_hooks.decode_merge_cap(scheduler, dec_cap)
    ran = []
    for request in running:
        if scheduler.current_step < request.next_decode_eligible_step:
            continue
        if dec_cap is not None and len(ran) >= dec_cap:
            continue
        request.next_decode_eligible_step = scheduler.current_step + pp_size
        ran.append(request)
    return ran


class Engine:
    """The step loop's order (executor_hooks.before_step_with_batch_queue: sync, then schedule), on the real driver."""

    def __init__(self, driver, decoders, *, step=0):
        self.driver = driver
        self.scheduler = types.SimpleNamespace(current_step=step, running=list(decoders))
        self.last_run = {r.request_id: None for r in decoders}
        self.max_wait = 0                 # the longest gap, in steps, between two runs of one live decoder
        self.trace = []

    def step(self, now_ns=None):
        if now_ns is None:
            self.driver.sync_decode_merge(self.scheduler)
        else:
            with mock.patch.object(so.time, 'monotonic_ns', lambda: now_ns):     # the observer's clock: 30 ms a step
                self.driver.sync_decode_merge(self.scheduler)
        ran = replay_step(self.scheduler, self.scheduler.running)
        step = self.scheduler.current_step
        for r in ran:
            r.num_output_tokens += 1
            last = self.last_run.get(r.request_id)
            if last is not None:
                self.max_wait = max(self.max_wait, step - last)
            self.last_run[r.request_id] = step
        for r in self.scheduler.running:
            last = self.last_run.get(r.request_id)
            if last is not None:
                self.max_wait = max(self.max_wait, step - last)          # a decoder not yet run again counts too
        self.trace.append(len(ran))
        self.scheduler.current_step += 1
        return ran

    def drop(self, count, how):
        """EOS: the last-run decoders leave (they produced their final token); cancel: the first ``count`` in list order."""
        running = self.scheduler.running
        if how == 'eos':
            order = sorted(running, key=lambda r: -(self.last_run.get(r.request_id) or -1))
        else:
            order = list(running)
        gone = order[:count]
        self.scheduler.running = [r for r in running if r not in gone]
        for r in gone:
            self.last_run.pop(r.request_id, None)


class DriverBase(c0ctl.Base):
    """test_stage1_step_engine's DriverBase (that module deletes its own after use): the real OverlapSegmentDriver."""

    async def asyncSetUp(self):
        with mock.patch.object(s1.ed, 'NativeSegmentDriver', OverlapSegmentDriver), \
                mock.patch.object(s1.sh, 'PrefillAdmissionController', so.StagedAdmissionController), \
                mock.patch.object(c0ctl, 'StagedSegmentDriver', OverlapSegmentDriver), \
                mock.patch.dict(os.environ, s1.STEP_ENV):
            await super().asyncSetUp()


class DriverCase(DriverBase):
    def phase_is(self, phase='native'):
        return mock.patch.object(self.driver, 'auto_arm', types.SimpleNamespace(phase=phase, manual_owned=False), create=True)

    def open_piece(self, rows):
        return mock.patch.object(self.driver, 'tx', None if rows is None else types.SimpleNamespace(chunk_start=1000, chunk_end=1000 + rows))


class Cohorts(DriverCase):
    async def test_off_runs_two_cohorts_through_the_native_phase(self):
        engine = Engine(self.driver, [Decoder() for _ in range(8)])
        self.driver.decode_merge = 'off'
        with self.phase_is('native'), self.open_piece(2048):
            for _ in range(12):
                engine.step()
        self.assertEqual(engine.trace, [4] * 12)
        self.assertEqual(engine.max_wait, 2)                                   # every decoder every other step
        self.assertFalse(getattr(engine.scheduler, 'flash_s5_decode_merge', False))
        self.assertEqual(self.driver.decode_step_counters, dict(lazy_marked=0, merge_steps=0, merge_on=0, merge_off=0))

    async def test_chunk_merges_the_same_decoders_into_even_steps(self):
        """The control: today's chunk mode on the same replay (all 8 on even steps, odd steps empty)."""
        engine = Engine(self.driver, [Decoder() for _ in range(8)])
        self.driver.decode_merge = 'chunk'
        with self.phase_is('native'), self.open_piece(2048):
            for _ in range(12):
                engine.step()
        self.assertEqual(engine.trace[2:], [8, 0] * 5)
        self.assertEqual(engine.max_wait, 2)

    async def test_off_small_follows_the_open_piece(self):
        engine = Engine(self.driver, [Decoder() for _ in range(8)])
        self.driver.decode_merge = 'off-small'
        with self.phase_is('native'):
            for rows, merged in ((None, False), (176, False), (1535, False), (1536, True), (2048, True), (100, False)):
                with self.open_piece(rows):
                    self.driver.sync_decode_merge(engine.scheduler)
                    self.assertIs(getattr(engine.scheduler, 'flash_s5_decode_merge', False), merged, rows)
        with self.phase_is('tail'), self.open_piece(2048):
            self.driver.sync_decode_merge(engine.scheduler)
            self.assertIs(engine.scheduler.flash_s5_decode_merge, False)       # outside the native phase: stock, as chunk
        c = self.driver.decode_step_counters
        self.assertEqual((c['merge_steps'], c['merge_on'], c['merge_off']), (2, 1, 1))

    async def test_off_small_alternates_bulk_and_small_pieces_and_loses_no_decoder(self):
        engine = Engine(self.driver, [Decoder() for _ in range(8)])
        self.driver.decode_merge = 'off-small'
        with self.phase_is('native'):
            for rows in (176, 2048, 176, 1536, None, 2048, 64):
                with self.open_piece(rows):
                    for _ in range(5):
                        engine.step()
        self.assertLessEqual(engine.max_wait, 3)                              # a merge from the odd cohort waits one step, once
        self.assertEqual(sum(engine.trace), sum(r.num_output_tokens - 1 for r in engine.scheduler.running))
        self.assertTrue(all(r.num_output_tokens >= 15 for r in engine.scheduler.running))

    async def test_chunk_mode_step_path_is_unchanged(self):
        """chunk, liveness off: the same trace and counters as the stage1-step suite's expectation, and no new attribute."""
        self.assertIsNone(self.driver.decode_liveness)
        engine = Engine(self.driver, [Decoder() for _ in range(8)])
        self.driver.decode_merge = 'chunk'
        with self.phase_is('idle'):
            for _ in range(4):
                engine.step()
        with self.phase_is('native'):
            for _ in range(6):
                engine.step()
        self.assertEqual(engine.trace, [4, 4, 4, 4, 4, 0, 8, 0, 8, 0])         # the odd cohort waits one step, once
        self.assertEqual(self.driver.decode_step_counters, dict(lazy_marked=0, merge_steps=6, merge_on=1, merge_off=0))
        self.assertNotIn('liveness', self.driver.stage_status()['overlap']['decode_step'])


class Shrinking(DriverCase):
    """Review item 9's sequence on the rule replay: 8 -> 4 -> 2 -> 1 -> 0, both parities, EOS and cancel, all modes."""

    async def asyncSetUp(self):
        with mock.patch.dict(os.environ, {'FLASH_S5_DECODE_LIVENESS': '1'}):
            await super().asyncSetUp()

    async def run_sequence(self, mode, start_step, how, rows, phase_flip):
        self.driver.decode_liveness = DecodeLiveness()
        self.driver.decode_merge = mode
        engine = Engine(self.driver, [Decoder(eligible=start_step) for _ in range(8)], step=start_step)
        now = [0]
        policy = types.SimpleNamespace(phase='native', manual_owned=False)

        def steps(count):
            for _ in range(count):
                now[0] += 30_000_000                                             # 30 ms a step
                engine.step(now_ns=now[0])
        with mock.patch.object(self.driver, 'auto_arm', policy, create=True), self.open_piece(rows):
            steps(6)
            for keep, offset in ((4, 0), (2, 1), (1, 0), (0, 1)):
                steps(offset)                                                    # the drop lands on the other parity every other time
                engine.drop(len(engine.scheduler.running) - keep, how)
                if phase_flip:
                    policy.phase = 'tail' if policy.phase == 'native' else 'native'
                steps(5)
                for r in engine.scheduler.running:
                    self.assertLessEqual(engine.max_wait, 3, (mode, start_step, how, keep))
                    self.assertGreater(r.num_output_tokens, 1)
            steps(3)
        self.assertEqual(engine.scheduler.running, [])
        self.assertEqual(engine.trace[-3:], [0, 0, 0])
        live = self.driver.decode_liveness.status()
        self.assertEqual(live['trips'], 0, (mode, start_step, how, live))
        self.assertEqual(live['live_max_stall_ms'], 0)
        return engine

    async def test_every_mode_both_parities_eos_and_cancel(self):
        for mode in ('off', 'off-small', 'chunk'):
            for start_step in (0, 1):
                for how in ('eos', 'cancel'):
                    for rows in (176, 2048):
                        for flip in (False, True):
                            engine = await self.run_sequence(mode, start_step, how, rows, flip)
                            self.assertLessEqual(engine.max_wait, 3)

    async def test_off_at_one_live_decoder_runs_it_every_other_step(self):
        engine = await self.run_sequence('off', 0, 'eos', 2048, False)
        # the stretch with one live decoder (after the third drop): one token every pp_size steps
        self.assertIn([1, 0, 1, 0], [engine.trace[i:i + 4] for i in range(len(engine.trace) - 3)])

    async def test_a_starving_cap_is_tripped(self):
        """Mutant: a cap of zero on every odd step without the even-step merge (the odd cohort never runs)."""
        self.driver.decode_liveness = DecodeLiveness(stall_ns=1_000_000_000)
        self.driver.decode_merge = 'off'
        engine = Engine(self.driver, [Decoder() for _ in range(4)])
        real = sh_hooks.decode_merge_cap
        starving = lambda scheduler, cap: 0 if scheduler.current_step % 2 else 1     # noqa: E731
        lines = []
        with mock.patch.object(sh_hooks, 'decode_merge_cap', starving), self.phase_is('native'), \
                mock.patch.object(so, '_progress_logger', lambda: types.SimpleNamespace(warning=lines.append)):
            for i in range(100):
                engine.step(now_ns=(i + 1) * 30_000_000)
        self.assertIs(sh_hooks.decode_merge_cap, real)
        live = self.driver.decode_liveness.status()
        self.assertEqual(live['trips'], 3)                                       # three of the four never advance; one trip each
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(line.startswith(so.LIVENESS_LINE) for line in lines))
        self.assertGreaterEqual(live['max_stall_ms'], 2900)


class Observer(unittest.TestCase):
    def scheduler(self, *requests):
        return types.SimpleNamespace(running=list(requests))

    def req(self, rid, out):
        return types.SimpleNamespace(request_id=rid, num_output_tokens=out)

    def test_quiet_while_decoders_advance_and_prefill_requests_are_ignored(self):
        obs = DecodeLiveness(stall_ns=100)
        a, b, p = self.req('a', 1), self.req('b', 3), self.req('p', 0)
        for t in range(0, 1000, 10):
            a.num_output_tokens += 1
            b.num_output_tokens += 1
            self.assertEqual(obs.check(self.scheduler(a, b, p), 'off', now_ns=t), [])
        self.assertEqual(obs.status()['trips'], 0)
        self.assertEqual(obs.status()['tracked_max'], 2)

    def test_one_trip_and_one_line_per_episode(self):
        obs = DecodeLiveness(stall_ns=100)
        a = self.req('a', 5)
        lines = []
        with mock.patch.object(so, '_progress_logger', lambda: types.SimpleNamespace(warning=lines.append)):
            obs.check(self.scheduler(a), 'off', now_ns=0)
            obs.check(self.scheduler(a), 'off', now_ns=50)
            self.assertEqual(obs.status()['trips'], 0)
            self.assertEqual(len(obs.check(self.scheduler(a), 'off', now_ns=150)), 1)
            obs.check(self.scheduler(a), 'off', now_ns=900)                    # same episode: no second trip
            a.num_output_tokens += 1
            obs.check(self.scheduler(a), 'off', now_ns=1000)                   # it moved: the episode ends
            obs.check(self.scheduler(a), 'off', now_ns=1200)                   # a second episode
        st = obs.status()
        self.assertEqual((st['trips'], len(lines)), (2, 2))
        self.assertEqual(st['live_max_stall_ms'], 0)
        self.assertIn('"mode":"off"', lines[0])

    def test_finished_requests_leave_the_table(self):
        obs = DecodeLiveness()
        obs.check(self.scheduler(self.req('a', 1), self.req('b', 1)), 'off', now_ns=0)
        obs.check(self.scheduler(self.req('b', 1)), 'off', now_ns=1)
        self.assertEqual(set(obs.seen), {'b'})

    def test_env_parse(self):
        env = {'FLASH_S5_DECODE_MERGE': '1', 'FLASH_S5_STAGE_OVERLAP': '1', 'FLASH_S5_STAGE_LOCAL': '1', 'FLASH_S5_STAGE_OVERLAP_DEFER_BIND': '1'}
        self.assertIs(so.decode_liveness_enabled({}), False)
        self.assertIs(so.decode_liveness_enabled(dict(env, FLASH_S5_DECODE_LIVENESS='0')), False)
        self.assertIs(so.decode_liveness_enabled(dict(env, FLASH_S5_DECODE_LIVENESS='1')), True)
        for bad in (dict(env, FLASH_S5_DECODE_LIVENESS='2'), {'FLASH_S5_DECODE_LIVENESS': '1'}):
            with self.assertRaises(so.DriverError):
                so.decode_liveness_enabled(bad)

    def test_the_observer_never_raises_into_the_step(self):
        class Broken:
            @property
            def running(self):
                raise RuntimeError('boom')
        driver = types.SimpleNamespace(decode_liveness=DecodeLiveness(), decode_merge='off', _merge_active=False,
                                       decode_step_counters=dict(lazy_marked=0, merge_steps=0, merge_on=0, merge_off=0),
                                       _decode_merge_wanted=lambda: False)
        OverlapSegmentDriver.sync_decode_merge(driver, Broken())
        self.assertEqual(driver.decode_liveness.counters['observer_errors'], 1)


class Liveness(DriverCase):
    async def asyncSetUp(self):
        with mock.patch.dict(os.environ, {'FLASH_S5_DECODE_LIVENESS': '1'}):
            await super().asyncSetUp()

    async def test_installed_by_the_switch_and_in_status(self):
        self.assertIsInstance(self.driver.decode_liveness, DecodeLiveness)
        block = self.driver.stage_status()['overlap']['decode_step']
        self.assertEqual(sorted(block['liveness']), ['checked_steps', 'live_max_stall_ms', 'max_stall_ms', 'stall_ms', 'tracked_max', 'trips'])
        self.assertEqual(block['liveness']['stall_ms'], 10_000)
        self.assertEqual(guard_mod.liveness_of({'native_stage_local': self.driver.stage_status()}), block['liveness'])

    async def test_the_observer_does_not_change_a_schedule(self):
        traces = []
        for installed in (True, False):
            self.driver.decode_liveness = DecodeLiveness() if installed else None
            for mode in ('chunk', 'off', 'off-small'):
                self.driver.decode_merge, self.driver._merge_active = mode, False
                engine = Engine(self.driver, [Decoder() for _ in range(8)])
                with self.phase_is('native'), self.open_piece(2048):
                    for _ in range(10):
                        engine.step()
                traces.append((mode, engine.trace))
        self.assertEqual(traces[:3], traces[3:])


class NoSwitch(DriverCase):
    async def test_default_has_no_observer_and_no_status_key(self):
        self.assertIsNone(self.driver.decode_liveness)
        self.assertNotIn('liveness', self.driver.stage_status()['overlap']['decode_step'])


class Control(DriverBase):
    ALL = dict(c0ctl.BOTH, **s1.STEP_ENV)
    C1 = dict(local=True, overlap=True, defer_bind=True)

    async def test_off_small_through_the_real_router(self):
        with process_env(self.ALL), mock.patch.dict('os.environ', s1.STEP_ENV):
            app, endpoint = attach(self.frontend, '1', '1')
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            for merge in ('off-small', 'off', 'chunk', 'off-small'):
                code, reply = await self.post(endpoint, app, 'native_stage_local', lazy_recv=True, decode_merge=merge, **self.C1)
                self.assertEqual(code, 200, reply)
                self.assertEqual(reply['decode_merge'], merge)
                self.assertEqual(self.driver.decode_merge, merge)
            code, reply = await self.post(endpoint, app, 'native_stage_local', lazy_recv=True, decode_merge='off-big', **self.C1)
            self.assertNotEqual(code, 200, reply)
            code, reply = await self.post(endpoint, app, 'native_stage_local', local=True, overlap=True, decode_merge='off-small')
            self.assertNotEqual(code, 200, reply)                                # needs the deferred bind, as chunk
            self.assertEqual(self.driver.decode_merge, 'off-small')
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)

    def test_value_sets(self):
        import native_overlap_control
        self.assertEqual(so.DECODE_MERGE_MODES, ('off', 'chunk', 'always', 'off-small'))
        text = Path(native_overlap_control.__file__).read_text()
        self.assertIn("decode_merge not in ('off', 'chunk', 'always', 'off-small')", text)


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def call(self, action, **fields):
        self.calls.append(action)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def status(trips, live=0):
    return {'native_stage_local': {'overlap': {'decode_step': {'liveness': dict(trips=trips, live_max_stall_ms=live, max_stall_ms=live)}}}}


class Guard(unittest.TestCase):
    def test_persist(self):
        row = {}
        guard_mod.persist(row, status(2))
        self.assertEqual(row['decode_liveness']['trips'], 2)
        row = {}
        guard_mod.persist(row, {'native_stage_local': {'overlap': {'decode_step': {}}}})
        guard_mod.persist(row, None)
        self.assertEqual(row, {})

    def test_an_engine_trip_ends_the_replay(self):
        g = guard_mod.Guard('nm.g2-live-0', clock=lambda: 100.0)
        client = guard_mod.WatchedClient(FakeClient([status(3), status(3, 4000), status(4, 11000)]), g)
        client.call('status')
        client.call('status')
        self.assertIsNone(g.trip)
        self.assertEqual(g.clock(), 100.0)
        client.call('status')
        self.assertEqual(g.trip[:2], ('ENGINE', 'engine-trip'))
        self.assertGreater(g.clock(), 100.0 + 10 ** 6)
        rec = dict(reasons=[], complete=True, statuses=[dict(label='before', decode_liveness=dict(trips=3)), dict(label='after', decode_liveness=dict(trips=4))])
        view = g.settle(rec)
        self.assertEqual(rec['reasons'], ['decode-liveness-guard:ENGINE:engine-trip'])
        self.assertIs(rec['complete'], False)
        self.assertEqual((view['trips_delta'], view['max_live_stall_ms'], view['ended_replay']), (1, 11000, True))

    def test_unanswered_polls(self):
        g = guard_mod.Guard('x', poll_fails=3)
        client = guard_mod.WatchedClient(FakeClient([OSError(), OSError(), status(0), OSError(), OSError(), OSError()]), g)
        for _ in range(6):
            with contextlib.suppress(OSError):
                client.call('status')
        self.assertEqual(g.trip[:2], ('ENGINE', 'status-unanswered'))
        self.assertEqual(g.failures, 5)

    def test_blind_guard_is_harness_and_does_not_end_the_replay(self):
        g = guard_mod.Guard('x', clock=lambda: 5.0)
        guard_mod.WatchedClient(FakeClient([{'native_stage_local': {}}]), g).call('status')
        self.assertIsNone(g.trip)
        self.assertEqual(g.clock(), 5.0)
        rec = dict(reasons=[], complete=True, statuses=[dict(label='before'), dict(label='after')])
        g.settle(rec)
        self.assertEqual(rec['reasons'], ['decode-liveness-guard:HARNESS:guard-blind'])

    def test_quiet_arm_is_unchanged(self):
        g = guard_mod.Guard('x')
        client = guard_mod.WatchedClient(FakeClient([status(0), status(0)]), g)
        client.call('status')
        client.call('status')
        rec = dict(reasons=[], complete=True, statuses=[dict(label='before', decode_liveness=dict(trips=0)), dict(label='after', decode_liveness=dict(trips=0))])
        g.settle(rec)
        self.assertEqual((rec['reasons'], rec['complete']), ([], True))
        self.assertEqual(rec['decode_liveness_guard']['trips_delta'], 0)

    def test_a_trip_only_in_after_minus_before_still_fails(self):
        g = guard_mod.Guard('x')
        rec = dict(reasons=[], complete=True, statuses=[dict(label='before', decode_liveness=dict(trips=0)), dict(label='after', decode_liveness=dict(trips=2))])
        g.settle(rec)
        self.assertEqual(rec['reasons'], ['decode-liveness-guard:ENGINE:engine-trip'])

    def test_watched_stream_passes_results_errors_and_the_thread_name(self):
        g = guard_mod.Guard('x')
        names = []

        def inner(url, body, key, origin, **kw):
            names.append(threading.current_thread().name)
            if body == 'bad':
                raise ValueError('turn error')
            return dict(ok=body)
        stream = guard_mod.watched_stream(inner, g, wait_s=0.01)
        out = {}
        t = threading.Thread(target=lambda: out.update(r=stream('u', 'b', None, 0.0)), name='compose4-agent-3')
        t.start()
        t.join()
        self.assertEqual(out['r'], dict(ok='b'))
        self.assertEqual(names, ['compose4-agent-3'])
        with self.assertRaises(ValueError):
            stream('u', 'bad', None, 0.0)

    def replay_with_a_stalled_stream(self, watched, stall_s):
        import compose4_replay as replay
        cfg = dict(replay.T.REPLAY, AGENTS=2, DURATION=60.0, WARMUP=0.0)
        release = threading.Event()

        def stalled(url, body, key, origin, **kw):
            release.wait(stall_s)
            raise OSError('released')
        g = guard_mod.Guard('nm.g2-live-0')
        stream = guard_mod.watched_stream(stalled, g, wait_s=0.05) if watched else stalled
        timer = threading.Timer(0.5, lambda: g._end('ENGINE', 'engine-trip'))
        timer.start()
        started = time.monotonic()
        try:
            _cfg, summary, turns, still_open = replay.run('http://127.0.0.1:9', 'm', None, 'run', 'nm.g2-live-0', cfg=cfg,
                                                          clock=g.clock, stream=stream,
                                                          sleep=lambda s: time.sleep(min(s, 0.05)), finish_bound_s=30.0)
        finally:
            elapsed = time.monotonic() - started
            release.set()
            timer.cancel()
        return elapsed, summary, turns, still_open

    def test_the_real_replay_loop_ends_within_seconds_on_a_trip(self):
        """compose4_replay.run with streams that do not return: the guard's trip ends it in about a second."""
        elapsed, summary, turns, still_open = self.replay_with_a_stalled_stream(True, 30.0)
        self.assertLess(elapsed, 5.0)
        self.assertEqual(summary['agents_died'], 0)
        self.assertEqual(len(turns) + len(still_open), 2)                     # each agent left its stalled turn (or is leaving)
        self.assertTrue(turns and all(r['error'].startswith('GuardEnded:decode-liveness-guard:ENGINE:engine-trip') for r in turns), turns)

    def test_mutant_without_the_watched_stream_waits_for_the_stalled_read(self):
        elapsed, summary, turns, still_open = self.replay_with_a_stalled_stream(False, 3.0)
        self.assertGreaterEqual(elapsed, 2.5)                                    # the clock jump alone does not free a blocked read

    def test_guarded_wraps_the_poll_client_and_the_run(self):
        seen = {}

        def replay_arm(a, row, client, poll_client, *, run=None):
            poll_client.call('status')
            seen['result'] = run('url', 'model', None, 'run', row['label'], cfg={}, stream=lambda *x, **k: None, sink=None)
            return dict(reasons=[], complete=True, statuses=[])

        def base_run(*args, **kwargs):
            seen['clock'] = kwargs['clock']
            seen['stream'] = kwargs['stream']
            return 'ran'
        arm = guard_mod.guarded(replay_arm, lambda: base_run)
        rec = arm(None, dict(label='n3.g2-live-0'), None, FakeClient([status(0)]))
        self.assertEqual(seen['result'], 'ran')
        self.assertEqual(seen['clock'].__self__.label, 'n3.g2-live-0')
        self.assertIn('decode_liveness_guard', rec)
        self.assertEqual(rec['decode_liveness_guard']['polls'], 1)
        box = types.SimpleNamespace(replay_arm=replay_arm, _box=types.SimpleNamespace(replay=types.SimpleNamespace(run=base_run)))
        first = guard_mod.install(box)
        self.assertIs(guard_mod.install(box), first)                           # idempotent


class SubmitCpu(unittest.TestCase):
    """Ride-along (b): thread CPU beside the wall stamps of flash_s5_submit, through the real worker RPC begin/return."""

    def run_rpcs(self, env):
        import executor_hooks as eh
        from native_measurement import encode_progress_meta, PROGRESS_META_KEY
        lines = []
        logger = types.SimpleNamespace(info=lambda line, *a: lines.append(line % a if a else line), warning=lambda *a: None)
        proc = types.SimpleNamespace(rank=2)
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(eh, '_WORKER_PROGRESS', None), \
                mock.patch.object(eh, '_SUBMIT_CPU', None), mock.patch.object(eh, '_ANATOMY', False), \
                mock.patch.object(eh, '_progress_logger', lambda: logger), mock.patch.object(eh, 'emit_receipt_lines', lambda *a: True):
            rpcs = []
            for seq, name in ((1, 'flash_s5_submit'), (2, 'execute_model'), (3, 'flash_s5_submit'), (4, 'flash_s5_release')):
                rpc = eh.worker_rpc_begin(proc, name, {PROGRESS_META_KEY: encode_progress_meta(seq, False)})
                if name == 'flash_s5_submit':
                    sum(range(20000))                                            # some thread CPU
                eh.worker_rpc_returned(proc, rpc, None)
                rpcs.append(rpc)
            acc = eh._submit_cpu()
        return rpcs, acc, [l for l in lines if l.startswith('flash-s5 submit-cpu ')]

    def test_off_by_default_the_rpc_record_is_unchanged(self):
        rpcs, acc, lines = self.run_rpcs({'FLASH_S5_NATIVE_PROGRESS': 'burst', 'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_SUBMIT_CPU': '0'})
        self.assertTrue(all(len(r) == 5 for r in rpcs))
        self.assertIsNone(acc)
        self.assertEqual(lines, [])

    def test_on_counts_submits_only_and_emits_one_line_per_release(self):
        import json as _json
        rpcs, acc, lines = self.run_rpcs({'FLASH_S5_NATIVE_PROGRESS': 'burst', 'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_SUBMIT_CPU': '1'})
        self.assertEqual([len(r) for r in rpcs], [6, 5, 6, 5])
        self.assertEqual(acc['n'], 2)
        self.assertGreater(acc['cpu_ns'], 0)
        self.assertGreaterEqual(acc['wall_ns'], acc['max_wall_ns'])
        self.assertEqual(sum(v[0] for v in acc['by_wall_ms'].values()), 2)
        self.assertEqual(len(lines), 1)
        row = _json.loads(lines[0][len('flash-s5 submit-cpu '):])
        self.assertEqual((row['rank'], row['n']), (2, 2))


J2_CONFIGURE = '''
import compose5_arms as table
ARM_TABLE = (table.SMOKE_ROW,) + tuple(('core', name, 'live', 0) for name in table.O3_PAIR) + (
    ('core', 'sp', 'live', 0), ('core', 'n3', 'live', 0),
    ('core', 'n3', 'live', 1), ('core', 'sp', 'live', 1))
def configure():
    table.ARM_TABLE = ARM_TABLE
    table.DECODE_ARMS = ()
    table.LAUNCH = dict(table.LAUNCH, per_bulk=8)
    table.LAUNCH_ENV = dict(table.LAUNCH_ENV, FLASH_S5_APPEND_YIELD_PER_BULK=8)
    table.LAUNCH_PER_BULK = 8
    table.REPLAY = dict(table.REPLAY, REQUEST_IDS=True)
    return table
'''


def j2_configure():
    """The bundle's n3j2_packet.configure when it is importable (the bundle's frozen run), else its text (the hub)."""
    try:
        import n3j2_packet
        return n3j2_packet.configure
    except ImportError:
        space = {}
        exec(compile(J2_CONFIGURE, 'n3j2_packet(inline)', 'exec'), space)
        return space['configure']


class Table(unittest.TestCase):
    def setUp(self):
        import compose5_arms
        self.saved = dict(vars(compose5_arms))
        self.module = compose5_arms

    def tearDown(self):
        for k in list(vars(self.module)):
            if k not in self.saved:
                delattr(self.module, k)
        for k, v in self.saved.items():
            setattr(self.module, k, v)

    def test_both_variants(self):
        import n3dmp_packet as dmp
        for variant, sel, mode in (('off', 'nm', 'off'), ('offsmall', 'ns', 'off-small')):
            self.tearDown()
            table = dmp.wrap(j2_configure(), variant)()
            table = dmp.wrap(j2_configure(), variant)()                          # again: j2's configure resets ARM_TABLE
            self.assertEqual(table.labels(), ['sm.g2-live-0', 'pf.g2-live-0', 'po.g2-live-0', 'pg.g2-live-0', 'sp.g2-live-0', 'n3.g2-live-0',
                                              sel + '.g2-live-0', sel + '.g2-live-1', 'n3.g2-live-1'])
            rows = table.by_label()
            self.assertEqual(rows[sel + '.g2-live-0']['decode_merge'], mode)
            self.assertEqual({table.replay_config(rows[k])['SEED'] for k in rows if rows[k]['kind'] == 'replay'}, {1})
            self.assertEqual(table.SELECTIONS.count(sel), 1)
            for name, value in vars(table).items():
                if isinstance(value, dict) and set(value) >= set(self.saved['SELECTIONS']):
                    self.assertIn(sel, value, name)                              # every per-selection table knows the probe

    def test_check_refuses_a_mutant(self):
        import n3dmp_packet as dmp
        table = dmp.wrap(j2_configure(), 'off')()
        table.DECODE_MERGE = dict(table.DECODE_MERGE, n3='off')
        with self.assertRaises(AssertionError):
            dmp.check(table, 'off')
        table.DECODE_MERGE = dict(table.DECODE_MERGE, n3='chunk')
        table.REPLAY_SEEDS = {0: 1, 1: 2}
        with self.assertRaises(AssertionError):
            dmp.check(table, 'off')


del DriverBase, DriverCase

if __name__ == '__main__':
    unittest.main()
