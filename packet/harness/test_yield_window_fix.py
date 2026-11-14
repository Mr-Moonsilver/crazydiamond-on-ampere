"""FLASH_S5_YIELD_WINDOW_FIX (lever-2 per-step brief, 14 Nov 2026): the Y2 window's staged-append re-check. CPU only.

Box evidence (w13/w14 records cxay12 b1-b4, cxl2off, cxl2on42; DESIGN-lever2-perstep.md): no window ever served a second
append (windows == appends in all six) and every staged next append ended 'next-left'. append_yield_window_begin runs
inside auto-arm's own PAUSED_ALL bracket (old_pause saved, PAUSED_ALL, arm, begin, old_pause restored);
_environment_refuses reads scheduler._pause_state, which stock set_pause_state writes (vllm v1 scheduler.py:2777), so
the re-check refused every staged append. The per-bulk lookup memo then kept it out of every later lookup until the bulk
ended (15-32 appends per record, 10-30 s each). The CPU rig's scheduler never updated _pause_state, so test_yield_window
could not see it: ``stock_pause`` below gives the rig stock's set_pause_state.

With the flag at 1 (needs FLASH_S5_YIELD_WINDOW=1): the begin's re-check reads the pause auto-arm restores right after
(policy.old_pause); every other gate stays, an operator/engine pause already in old_pause still refuses; a staged append
still refused leaves the per-bulk memo once per request per bulk (re-offered at the next chunk boundary).
Rigs: test_yield_window.window_rig (test_append_yield.Rig: real controller, segment driver, AutoArmPolicy, four bridge
workers) and test_lane_o3.yield_rig. No GPU, CUDA or box claim.
"""
import inspect
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'patches/s5_engine')]
import test_append_yield as ay  # noqa: E402
import test_lane_o3 as lo3  # noqa: E402
import test_tail_select as ts  # noqa: E402
import test_yield_window as yw  # noqa: E402
import engine_driver as ed  # noqa: E402
import scheduler_hooks as sh  # noqa: E402
from e3_types import E3ContractError  # noqa: E402

NS = types.SimpleNamespace
Pause = ay.Pause
OTHER_UID, THIRD_UID = yw.OTHER_UID, yw.THIRD_UID
PAUSED_NEW = NS(name='PAUSED_NEW')     # leaf: another pinned PauseState member (the gate reads .name)


def stock_pause(rig):
    """Leaf: the pinned stock Scheduler.set_pause_state (v1 scheduler.py:2777-2779) writes _pause_state, which the
    pause_state property returns; the rig's own setter (test_e3_native_overlap.core_for) keeps driving core.paused."""
    s = rig.s
    rig_set = s.set_pause_state

    def set_pause_state(state):
        rig_set(state)
        s._pause_state = state
        s.pause_state = state
    s.set_pause_state = set_pause_state
    s._pause_state = s.pause_state
    return rig


def fix_rig(fix=True, **kw):
    rig = stock_pause(yw.window_rig(**kw))
    rig.c.yield_window_fix = fix
    return rig


def stranded_after_resume(tc, rig, other):
    """The owner re-entered; its next chunk commits; at that junction ``other`` is not taken (the memo holds it)."""
    rig.admit()
    rig.finish_chunk()
    y = rig.d._yield
    tc.assertTrue(y is None or y['append'] is not other, 'the stranded append got a yield')
    tc.assertIn(other, rig.s.skipped_waiting.items)


class Setting(unittest.TestCase):
    def controller(self, env):
        base = {k: v for k, v in os.environ.items() if k not in (sh.YIELD_WINDOW_ENV, sh.YIELD_WINDOW_FIX_ENV)}
        base.update(env)
        with mock.patch.dict(os.environ, base, clear=True):
            scheduler = NS(kv_cache_manager=NS(), max_num_scheduled_tokens=1024)
            return sh.PrefillAdmissionController(scheduler, leases=sh.SchedulerLeases(), frontier=ay.adm.ah.CommittedFrontier(),
                                                 engine_epoch=7, chunk_tokens=4, ranks=(0, 1), max_history=64)

    def test_strict_default_off_and_needs_the_window(self):
        self.assertFalse(self.controller({}).yield_window_fix)
        self.assertFalse(self.controller({sh.YIELD_WINDOW_ENV: '1', sh.YIELD_WINDOW_FIX_ENV: '0'}).yield_window_fix)
        self.assertTrue(self.controller({sh.YIELD_WINDOW_ENV: '1', sh.YIELD_WINDOW_FIX_ENV: '1'}).yield_window_fix)
        with self.assertRaisesRegex(E3ContractError, 'needs FLASH_S5_YIELD_WINDOW=1'):
            self.controller({sh.YIELD_WINDOW_FIX_ENV: '1'})
        for bad in ('2', 'yes', '', ' 1', 'true'):
            with self.subTest(bad=bad), self.assertRaises(E3ContractError):
                self.controller({sh.YIELD_WINDOW_ENV: '1', sh.YIELD_WINDOW_FIX_ENV: bad})

    def test_status_block_only_with_the_flag(self):
        off = self.controller({sh.YIELD_WINDOW_ENV: '1'}).native_admission_record()['queue_policy']
        self.assertNotIn('yield_window_fix', off)
        on = self.controller({sh.YIELD_WINDOW_ENV: '1', sh.YIELD_WINDOW_FIX_ENV: '1'}).native_admission_record()
        self.assertEqual(on['queue_policy']['yield_window_fix'],
                         dict(begins=0, own_pause_passed=0, still_refused=0, reoffered=0))

    def test_environment_reads_the_given_pause_only_when_given(self):
        c = self.controller({})
        for own, given, want in ((Pause.PAUSED_ALL, sh._OWN_PAUSE, True), (Pause.UNPAUSED, sh._OWN_PAUSE, False),
                                 (Pause.PAUSED_ALL, Pause.UNPAUSED, False), (Pause.UNPAUSED, PAUSED_NEW, True),
                                 (Pause.PAUSED_ALL, None, False)):
            s = NS(connector=None, _pause_state=own, vllm_config=None)
            with self.subTest(own=own, given=given):
                self.assertIs(c._environment_refuses(s, given), want)
        s = NS(connector=object(), _pause_state=Pause.UNPAUSED, vllm_config=None)
        self.assertTrue(c._environment_refuses(s, Pause.UNPAUSED), 'a connector still refuses')
        s = NS(connector=None, _pause_state=Pause.UNPAUSED, vllm_config=NS(speculative_config=object()))
        self.assertTrue(c._environment_refuses(s, Pause.UNPAUSED), 'speculation still refuses')


class Window(unittest.TestCase):
    """The FULL-window path with stock's pause: the defect with the flag off, the second append served with it on."""

    def setUp(self):
        self.log = yw.Lines(self)

    def test_off_the_staged_append_is_refused_and_stranded(self):
        """The box's as-built behaviour (flag 0): next-left by the environment gate, then no yield for the bulk."""
        rig = fix_rig(fix=False)
        append, (other,) = yw.served_first(rig)
        ay.to_resume(rig)
        w = rig.d._yield['window']
        self.assertEqual(w['end'], 'next-left')
        self.assertGreaterEqual(rig.c.yield_refusals.get('environment', 0), 1)
        self.assertEqual(yw.counts(rig)['next_left'], 1)
        stranded_after_resume(self, rig, other)
        ay.finish_bulk(rig)
        self.assertEqual(rig.d.append_yield_counters['admitted'], 1, 'one append served for the whole bulk')
        self.assertEqual(rig.c.yield_window_fix_counts, dict(begins=0, own_pause_passed=0, still_refused=0, reoffered=0))
        self.assertNotIn('yield_window_fix', rig.c.native_admission_record()['queue_policy'])

    def test_on_the_staged_append_begins_in_the_window(self):
        rig = fix_rig()
        d = rig.d
        append, (other,) = yw.served_first(rig)
        frontier = rig.bulk.num_computed_tokens
        yw.to_window_append(rig)
        self.assertIs(d._yield['append'], other)
        self.assertEqual(d._yield['frontier'], frontier)
        self.assertEqual(rig.c.yield_window_fix_counts, dict(begins=1, own_pause_passed=1, still_refused=0, reoffered=0))
        ay.run_append(rig)
        ay.to_full(rig)
        rig.stock_handback(other)
        ay.to_resume(rig)
        ay.finish_bulk(rig)
        w = yw.counts(rig)
        self.assertEqual((w['windows'], w['appends'], w['reentries_avoided'], w['next_left']), (1, 2, 1, 0))
        self.assertEqual(w['ends'], {'none-waiting': 1})
        self.assertEqual(d.append_yield_counters['admitted'], 2)
        self.assertEqual(rig.c.native_admission_record()['queue_policy']['yield_window_fix']['own_pause_passed'], 1)
        self.assertEqual(rig.s._pause_state, Pause.UNPAUSED, 'auto-arm restored its pause')

    def test_on_three_appends_fcfs_in_one_window(self):
        rig = fix_rig()
        append, held = yw.served_first(rig, (OTHER_UID, THIRD_UID))
        order = []
        for _ in range(2):
            yw.to_window_append(rig)
            current = rig.d._yield['append']
            order.append(current.request_id)
            ay.run_append(rig)
            ay.to_full(rig)
            rig.stock_handback(current)
        ay.to_resume(rig)
        ay.finish_bulk(rig)
        self.assertEqual(order, [OTHER_UID, THIRD_UID])
        self.assertEqual(yw.counts(rig)['per_window'], {'3': 1})
        self.assertEqual(rig.c.yield_window_fix_counts['own_pause_passed'], 2)

    def test_on_the_begin_reads_auto_arms_restored_pause(self):
        rig = fix_rig()
        seen = []
        real = rig.c.append_yield_still_held

        def spy(scheduler, request, decode_counts=None, pause_state=sh._OWN_PAUSE):
            seen.append((pause_state, scheduler._pause_state, rig.p.old_pause))
            return real(scheduler, request, decode_counts, pause_state)
        append, (other,) = yw.served_first(rig)
        with mock.patch.object(rig.c, 'append_yield_still_held', spy):
            yw.to_window_append(rig)
        begin = [s for s in seen if s[0] is not sh._OWN_PAUSE]
        self.assertEqual(len(begin), 1)
        self.assertEqual(begin[0], (Pause.UNPAUSED, Pause.PAUSED_ALL, Pause.UNPAUSED),
                         "the begin runs under auto-arm's PAUSED_ALL and reads the restored UNPAUSED")

    def test_on_an_operator_pause_in_old_pause_still_refuses(self):
        rig = fix_rig()
        append, (other,) = yw.served_first(rig)
        policy = NS(old_pause=PAUSED_NEW, decode_counts=None)
        rig.s._pause_state = Pause.PAUSED_ALL
        self.assertFalse(rig.d._yield_window_fixed_held(other, policy, None))
        self.assertEqual(rig.c.yield_window_fix_counts, dict(begins=1, own_pause_passed=0, still_refused=1, reoffered=0))
        no_saved = NS(old_pause=None, decode_counts=None)
        self.assertFalse(rig.d._yield_window_fixed_held(other, no_saved, None), 'no saved pause: the scheduler\'s own')
        rig.s._pause_state = Pause.UNPAUSED
        self.assertTrue(rig.d._yield_window_fixed_held(other, no_saved, None))

    def test_on_a_refused_begin_re_offers_once_at_the_next_junction(self):
        """Another refusal at the begin (here forced once): next-left as built, then the memo entry is dropped and the
        next chunk boundary selects the append (as built it waits for the bulk's end)."""
        rig = fix_rig()
        append, (other,) = yw.served_first(rig)
        real = ed.NativeSegmentDriver._yield_window_fixed_held
        calls = []

        def once(self, candidate, policy, counts):
            calls.append(candidate.request_id)
            if len(calls) == 1:
                self.controller.yield_window_fix_counts['begins'] += 1
                self.controller.yield_window_fix_counts['still_refused'] += 1
                return False
            return real(self, candidate, policy, counts)
        with mock.patch.object(ed.NativeSegmentDriver, '_yield_window_fixed_held', once):
            ay.to_resume(rig)
            self.assertEqual(rig.d._yield['window']['end'], 'next-left')
            self.assertEqual(rig.c.yield_window_fix_counts['reoffered'], 1)
            key = rig.d._yield['key']
            self.assertEqual(rig.d._yield_reoffered, (key, {OTHER_UID}))
            self.assertNotIn(OTHER_UID, rig.d._yield_scan[1])
            rig.admit()
            rig.finish_chunk()
            self.assertEqual(rig.d.append_yield_phase(), 'closing', 'a yield for the re-offered append at frontier 8')
            self.assertIs(rig.d._yield['append'], other)
        self.assertEqual(rig.d.append_yield_counters['selected'], 2)

    def test_the_re_offer_is_once_per_request_per_bulk(self):
        rig = fix_rig()
        append, (other,) = yw.served_first(rig)
        d = rig.d
        key = d._yield['key']
        d._yield_scan = (key, {OTHER_UID})
        y = dict(key=key)
        self.assertTrue(d._yield_window_reoffer(y, other))
        d._yield_scan[1].add(OTHER_UID)
        self.assertFalse(d._yield_window_reoffer(y, other), 'second time in the same bulk: kept in the memo')
        self.assertIn(OTHER_UID, d._yield_scan[1])
        self.assertFalse(d._yield_window_reoffer(dict(key=('another-bulk', 1)), other), 'a memo of another bulk')
        self.assertEqual(rig.c.yield_window_fix_counts['reoffered'], 1)
        d._yield_scan = (('next-bulk', 2), {OTHER_UID})
        self.assertTrue(d._yield_window_reoffer(dict(key=('next-bulk', 2)), other), 'a new bulk: once again')


class WindowStaged(ay.StagedDriver, Window):
    pass


class O3Window(unittest.TestCase):
    """O3 installed and selected, stock's pause: the census arms the window's next append on the kept split route."""

    def setUp(self):
        self.log = ts.Log()
        patcher = mock.patch.object(ed, '_progress_logger', lambda: self.log)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_o3(self, fix):
        rig = lo3.yield_rig()
        stock_pause(rig)
        rig.c.yield_window = True
        rig.c.yield_window_fix = fix
        rig.s.max_num_running_reqs = 8
        p, d = rig.p, rig.d
        append = ay.run_to_yield(rig)
        other = rig.add_held(OTHER_UID, 6)
        ay.to_append(rig)
        ay.run_append(rig)
        ay.policy_ticks(rig, 1)
        lo3.split_handback_step(rig, append)
        lo3.worker_lag(rig, append)
        ay.policy_ticks(rig, 1)
        lo3.first_decode_step(rig, append)
        ay.policy_ticks(rig, 1)
        self.assertIs(d._yield['window']['next'], other)
        ts.landed(rig, append)
        return rig, other

    def test_off_next_left_on_the_o3_path(self):
        rig, other = self.run_o3(False)
        rig.tick(lambda: rig.d.append_yield_phase() == 'resume')
        self.assertEqual(rig.d._yield['window']['end'], 'next-left')

    def test_on_the_next_append_begins_on_the_kept_split(self):
        rig, other = self.run_o3(True)
        generation = rig.d.route_generation
        rig.tick(lambda: rig.d.append_yield_phase() == 'append')
        self.assertIs(rig.d._yield['append'], other)
        self.assertTrue(rig.d._split_ready)
        self.assertEqual(rig.d.route_generation, generation)
        self.assertEqual(rig.c.yield_window_fix_counts['own_pause_passed'], 1)


class OffIsUnchanged(unittest.TestCase):
    """Flag at 0: the begin calls the parent's re-check (no pause_state argument), never the fix helpers."""

    def setUp(self):
        yw.Lines(self)

    def test_off_never_calls_the_fix(self):
        boom = mock.Mock(side_effect=AssertionError('fix call with the flag off'))
        with mock.patch.object(ed.NativeSegmentDriver, '_yield_window_fixed_held', boom), \
                mock.patch.object(ed.NativeSegmentDriver, '_yield_window_reoffer', boom):
            rig = fix_rig(fix=False)
            yw.served_first(rig)
            ay.to_resume(rig)
            ay.finish_bulk(rig)
        rig = yw.window_rig()                                 # the parent's rig (no stock pause): two appends
        yw.served_first(rig)
        with mock.patch.object(ed.NativeSegmentDriver, '_yield_window_fixed_held', boom):
            yw.to_window_append(rig)

    def test_off_begin_source_is_the_parents_call(self):
        source = inspect.getsource(ed.NativeSegmentDriver.append_yield_window_begin)
        self.assertIn("held = self.controller.append_yield_still_held(self.scheduler, candidate, counts)\n", source)
        self.assertEqual(source.count('pause_state'), 0, 'the off branch passes no pause')


class Mutants(unittest.TestCase):
    """Each rule's mutant must fail its test (the method's own source, mutated, bound in its module)."""

    def setUp(self):
        yw.Lines(self)

    run_mutant = yw.Mutants.run_mutant
    case = yw.Mutants.case

    def test_begin_reads_its_own_pause_caught(self):
        self.run_mutant(ed.NativeSegmentDriver, '_yield_window_fixed_held',
                        "held = c.append_yield_still_held(self.scheduler, candidate, counts, pause_state=restored)\n",
                        "held = c.append_yield_still_held(self.scheduler, candidate, counts)\n",
                        self.case(Window, 'test_on_the_staged_append_begins_in_the_window'))

    def test_begin_ignores_any_pause_caught(self):
        self.run_mutant(ed.NativeSegmentDriver, '_yield_window_fixed_held',
                        "pause_state=restored)\n", "pause_state=None)\n",
                        self.case(Window, 'test_on_an_operator_pause_in_old_pause_still_refuses'))

    def test_unbounded_re_offer_caught(self):
        self.run_mutant(ed.NativeSegmentDriver, '_yield_window_reoffer',
                        "        if uid in done[1]:\n            return False\n", "",
                        self.case(Window, 'test_the_re_offer_is_once_per_request_per_bulk'))

    def test_no_re_offer_caught(self):
        self.run_mutant(ed.NativeSegmentDriver, '_yield_window_reoffer',
                        "        scan[1].discard(uid)\n", "",
                        self.case(Window, 'test_on_a_refused_begin_re_offers_once_at_the_next_junction'))


if __name__ == '__main__':
    unittest.main()
