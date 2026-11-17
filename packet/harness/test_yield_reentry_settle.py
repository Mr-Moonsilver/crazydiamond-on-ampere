"""yield-reentry (lead, 12 Nov 2026): the yield re-entry barrier diagnostic and FLASH_S5_YIELD_REENTRY_SETTLE. CPU only.

Box evidence (V1 j2 records, builder-tail-route-20261112/STATUS-lever2.md): 223 of 242 FULL yield re-entries armed after
exactly the 4 refused barriers REENTRY_RETRIES allows; a 5th refusal ends the yield and sends the bulk to stock prefill.
The records cannot say which proof field refused.

  Diagnostic  always on, observational: every refused re-entry proof (drain cohort, barrier, identity) is counted by
              stage and failing field with one 'flash-s5 yield-reentry proof-miss' line (retry index, the bound, ms since
              the re-entry began); the retries an armed re-entry needed are a histogram. Inert: decisions, controls,
              journal, pauses and phases are those of the same run with the diagnostic removed, flag 0 and flag 1.
  Settle      FLASH_S5_YIELD_REENTRY_SETTLE=1: the FULL window waits, unpaused and holding as today, for the returned
              append's first decode to be scheduled (the run-end path's K2 rule), then today's re-entry; bounded by
              O3_TAIL_TICKS, past it today's path; the retry bound and the stock fallback unchanged.

Rig: test_append_yield.Rig (real PrefillAdmissionController, segment driver, AutoArmPolicy, four bridge workers). The
stock steps of the append's handback are leaves, split as the box runs them (the worker counts a row's prefill from the
step that carries it, the scheduler's pre-advance count):
  handback_scheduled  the last prompt row scheduled: scheduler RUNNING, computed = prompt, no token yet; every worker
                      row prefill_len = prompt, computed prefill = prompt - 1 (the census reads it as prefill);
  handback_landed     its output processed: one sampled token;
  decode_scheduled    the first decode scheduled: scheduler computed = prompt + 1; every worker row computed = prompt.
No GPU, CUDA or box claim: which field refuses on the box, and the time a settle saves, are the box run's.
"""
import io
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'patches/s5_engine')]
import test_append_yield as ay  # noqa: E402
import native_auto_arm as auto  # noqa: E402
import scheduler_hooks as sh  # noqa: E402
from e3_types import E3ContractError  # noqa: E402

NS = types.SimpleNamespace
Rig = ay.Rig
DIAG_METHODS = ('_yrs_refusal', '_yrs_barrier_pass', '_yrs_armed')


# -- stock leaves (see the module docstring) ----------------------------------------------------------------------
def handback_scheduled(rig, append):
    for queue in (rig.s.skipped_waiting, rig.s.waiting):
        if append in queue.items:
            queue.items.remove(append)
    append.num_computed_tokens = append.num_tokens
    append.status = NS(name='RUNNING')
    rig.s.running.append(append)
    for worker in rig.stack.workers.values():
        states = worker.model_runner.req_states
        slot = len(states.prefill_len.np)
        states.req_id_to_index[append.request_id] = slot
        states.prefill_len.np.append(append.num_prompt_tokens)
        states.num_computed_prefill_tokens.append(append.num_prompt_tokens - 1)


def handback_landed(rig, append):
    append.num_output_tokens = 1


def decode_scheduled(rig, append):
    append.num_tokens = append.num_prompt_tokens + 1
    append.all_token_ids = list(range(append.num_tokens))
    append.num_computed_tokens = append.num_prompt_tokens + 1
    for worker in rig.stack.workers.values():
        states = worker.model_runner.req_states
        states.num_computed_prefill_tokens[states.req_id_to_index[append.request_id]] = append.num_prompt_tokens


def window(rig):
    return ay.handback_window(rig)


def yrs(rig):
    return rig.d.append_yield_record()['declines']['yield_reentry']


class Lines:
    """The progress logger's lines (engine_driver._progress_logger), captured."""

    def __init__(self):
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger = logging.getLogger('test-yield-reentry')
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.patch = mock.patch('engine_driver._progress_logger', lambda: self.logger)

    def __enter__(self):
        self.patch.start()
        return self

    def __exit__(self, *exc):
        self.patch.stop()
        self.logger.removeHandler(self.handler)

    def misses(self):
        return [line for line in self.stream.getvalue().splitlines() if 'yield-reentry proof-miss' in line]


class Trace:
    """Every decision the policy makes, for the inertness comparison: per tick the policy phase, the yield phase, the
    scheduler pause, the route; every control action submitted; the journal kinds and reasons at the end."""

    def __init__(self, rig):
        self.rig, self.ticks, self.controls = rig, [], []
        real = rig.d.submit_control

        def submit(command):
            self.controls.append(command.get('action') if type(command) is dict else command)
            return real(command)
        rig.d.submit_control = submit

    def tick(self, count=1):
        rig = self.rig
        for _ in range(count):
            ay.policy_ticks(rig, 1)
            self.ticks.append((rig.p.phase, rig.d.append_yield_phase(), str(rig.s.pause_state), rig.d._split_ready,
                               rig.d.route_generation, rig.p.reentry_refusals))

    def result(self):
        journal = [(e['kind'], e.get('reason'), e.get('stage'), e.get('passes')) for e in self.rig.d._journal]
        counters = dict(self.rig.d.append_yield_counters)
        return dict(ticks=self.ticks, controls=self.controls, journal=journal, counters=counters,
                    restores=dict(self.rig.p.restores), rollbacks=dict(self.rig.p.rollbacks))


def scenario_refused_then_armed(settle, decode_after=2, retries=4):
    """The box's shape: the handback scheduled and landed, the barrier refused while the workers read it as prefill,
    the first decode scheduled after ``decode_after`` scheduler passes, then the re-entry arms and the bulk reopens."""
    rig = Rig(cached={}, reentry_retries=retries)
    rig.p.yield_settle = settle
    append = window(rig)
    trace = Trace(rig)
    handback_scheduled(rig, append)
    trace.tick()
    handback_landed(rig, append)
    passes = 0
    for _ in range(4 * auto.O3_TAIL_TICKS):
        if rig.d.append_yield_phase() in ('resume', None):
            break
        if passes == decode_after:
            decode_scheduled(rig, append)
        ay.step_admit(rig)                       # one scheduler pass (nothing admitted: the bulk is held)
        passes += 1
        trace.tick()
    return rig, trace, append


class Setting(unittest.TestCase):
    def test_parse_is_strict_and_defaults_off(self):
        self.assertFalse(auto.yield_settle_enabled({}))
        self.assertFalse(auto.yield_settle_enabled({'FLASH_S5_YIELD_REENTRY_SETTLE': '0'}))
        self.assertTrue(auto.yield_settle_enabled({'FLASH_S5_YIELD_REENTRY_SETTLE': '1'}))
        for bad in ('2', 'yes', '', 'true', ' 1'):
            with self.subTest(bad=bad), self.assertRaises(E3ContractError):
                auto.yield_settle_enabled({'FLASH_S5_YIELD_REENTRY_SETTLE': bad})
        self.assertIs(auto.AutoArmPolicy.yield_settle, False)

    def test_install_sets_the_policy_flag(self):
        for value, expected in (('0', False), ('1', True), (None, False)):
            with self.subTest(value=value):
                rig = Rig(cached={})
                rig.d.auto_arm = None
                rig.core._flash_s5_startup_ready = True
                env = {'FLASH_S5_AUTO_ARM': '1'}
                if value is not None:
                    env['FLASH_S5_YIELD_REENTRY_SETTLE'] = value
                self.assertIs(auto.install(rig.core, rig.d, env).yield_settle, expected)


class Diagnostic(unittest.TestCase):
    """(1) always on, observational."""

    def test_a_prefill_reading_barrier_is_named_counted_and_logged(self):
        with Lines() as lines:
            rig, _trace, _append = scenario_refused_then_armed(settle=False, decode_after=2)
        block = yrs(rig)
        self.assertEqual(block['schema'], 's5-yield-reentry/1')
        self.assertEqual((block['settle_on'], block['retry_bound']), (0, 4))
        self.assertEqual(rig.d.append_yield_phase(), 'resume')
        refused = block['refused']
        self.assertEqual(set(refused), {'barrier:prefill-requests'})
        n = refused['barrier:prefill-requests']
        self.assertGreaterEqual(n, 1)
        self.assertEqual(block['refused_retry'], {str(i): 1 for i in range(1, n + 1)})
        self.assertEqual(block['retries_until_armed'], {str(n): 1})
        self.assertEqual((block['barrier_pass'], block['barrier_first_try'], block['exceeded']), (1, 0, 0))
        self.assertEqual(block['armed_ms'][0], 1)
        self.assertEqual(block['errors'], 0)
        misses = lines.misses()
        self.assertEqual(len(misses), n)
        self.assertIn('stage=barrier field=prefill-requests', misses[0])
        self.assertIn('got=1 want=0 retry=1 bound=4 ends=0', misses[0])
        # night-pack fix 6: the returned append's state at each refusal (decode scheduled, output tokens, running)
        self.assertRegex(misses[0], r'elapsed_ms=\d+\.\d settle=0 append_dec=0 append_out=\d+ running=\d+$')
        self.assertTrue(all(' append_dec=0 ' in m for m in misses), misses)
        for text in misses + [repr(block)]:
            self.assertNotIn(ay.APPEND_UID, text)
            self.assertNotIn(ay.BULK_UID, text)

    def test_a_refusal_past_the_bound_is_counted_and_ends_as_today(self):
        with Lines() as lines:
            rig, _trace, _append = scenario_refused_then_armed(settle=False, decode_after=99, retries=2)
        block = yrs(rig)
        self.assertEqual(block['exceeded'], 1)
        self.assertEqual(block['retries_until_armed'], {'exceeded': 1})
        self.assertEqual(block['refused'], {'barrier:prefill-requests': 3})
        self.assertEqual(rig.journal('append_yield_reentry_refused')[-1]['passes'], 3)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'reentry-refused')
        self.assertIn('ends=1', lines.misses()[-1])

    def test_inert_against_the_same_run_without_it_flag_off_and_on(self):
        for settle in (False, True):
            for decode_after in (0, 2, 99):
                with self.subTest(settle=settle, decode_after=decode_after):
                    _rig, real, _a = scenario_refused_then_armed(settle=settle, decode_after=decode_after)
                    with mock.patch.multiple(auto.AutoArmPolicy, **{m: lambda self, *a, **k: None for m in DIAG_METHODS}), \
                            mock.patch.object(auto.AutoArmPolicy, '_yrs_proof_why', lambda self, *a, **k: None), \
                            mock.patch.object(auto.AutoArmPolicy, '_yrs_cohort_why', lambda self, *a, **k: None):
                        _rig2, bare, _b = scenario_refused_then_armed(settle=settle, decode_after=decode_after)
                    self.assertEqual(real.result(), bare.result())

    def test_no_extra_control_or_rpc(self):
        _rig, trace, _a = scenario_refused_then_armed(settle=False, decode_after=2)
        refusals = yrs(_rig)['refused']['barrier:prefill-requests']
        # every refusal is one barrier; the pass is barrier, identity, enter_split; nothing else was sent
        self.assertEqual(trace.controls, ['barrier'] * (refusals + 1) + ['identity', 'enter_split'])

    def test_proof_why_is_none_exactly_when_proof_holds(self):
        """Checked at the moment the real proof runs (route, generation and cohort as the policy holds them then)."""
        import copy
        rig = Rig(cached={})
        append = window(rig)
        handback_scheduled(rig, append)
        handback_landed(rig, append)
        decode_scheduled(rig, append)
        real = auto.AutoArmPolicy._proof
        checked = []

        def inv(r, i=0):
            return r['workers'][i]['native_inventory']['auto_arm_decode']
        common = [
            ('worker-rows', lambda r: r['workers'].pop()),
            ('worker-ranks', lambda r: r['workers'][0].__setitem__('worker_rank', 99)),
            ('engine-epoch', lambda r: r['workers'][1].__setitem__('engine_epoch', -1)),
            ('route', lambda r: r['workers'][2].__setitem__('route', 'split')),
            ('route-generation', lambda r: r['workers'][3].__setitem__('route_generation', -5)),
            ('prefill-requests', lambda r: r['workers'][0].__setitem__('prefill_requests', 1)),
            ('live-unavailable', lambda r: r['workers'][0]['live_decode'].__setitem__('available', False)),
            ('live-uids', lambda r: r['workers'][1]['live_decode'].__setitem__('live_request_uids', [])),
            ('decode-eligible', lambda r: r['workers'][2].__setitem__('decode_eligible', 7)),
        ]
        extra = {True: [('capability-counts', lambda r: inv(r).__setitem__('supported_counts', [0])),
                        ('capability-identity', lambda r: inv(r, 1).__setitem__('schema', 'x')),
                        ('capability-cover', lambda r: inv(r, 2).__setitem__('supported_counts', [2, 3]))],
                 False: [('barrier-clean', lambda r: r['workers'][0].__setitem__('barrier_clean', False)),
                         ('drained-streams', lambda r: r['workers'][3].__setitem__('drained_streams', 0))]}

        def capture(self, result, *, capability=False):
            base = copy.deepcopy(result)
            saved = self.decode_counts
            ok = real(self, copy.deepcopy(base), capability=capability)
            checked.append(('base', capability, ok, self._yrs_proof_why(copy.deepcopy(base), capability=capability)[0]))
            for field, mutate in common + extra[capability]:
                mutated = copy.deepcopy(base)
                mutate(mutated)
                checked.append((field, capability, real(self, copy.deepcopy(mutated), capability=capability),
                                self._yrs_proof_why(mutated, capability=capability)[0]))
            checked.append(('worker-rows', capability, False, self._yrs_proof_why(None, capability=capability)[0]))
            self.decode_counts = saved
            return real(self, result, capability=capability)
        with mock.patch.object(auto.AutoArmPolicy, '_proof', capture):
            ay.to_resume(rig)
        self.assertEqual({c[1] for c in checked}, {False, True})
        for field, capability, ok, why in checked:
            with self.subTest(field=field, capability=capability):
                if field == 'base':
                    self.assertTrue(ok)
                    self.assertEqual(why, 'none')
                else:
                    self.assertFalse(ok)
                    self.assertEqual(why, field)

    def test_cohort_why_matches_reentry_cohort(self):
        rig = Rig(cached={})
        p = rig.p
        self.assertEqual(p._yrs_cohort_why()[0], 'none')
        self.assertIsNotNone(p._reentry_cohort())
        prefilling = ay.Req('ra-prefill', 9, computed=4)
        prefilling.status = NS(name='RUNNING')
        rig.s.running.append(prefilling)
        self.assertIsNone(p._reentry_cohort())
        self.assertEqual(p._yrs_cohort_why()[:3], ('prefill', None, 1))
        prefilling.num_computed_tokens = 9
        prefilling.lora_request = object()
        self.assertIsNone(p._reentry_cohort())
        self.assertEqual(p._yrs_cohort_why()[0], 'lora')

    def test_flag_off_never_enters_the_settle(self):
        with mock.patch.object(auto.AutoArmPolicy, '_yrs_settle_waits', side_effect=AssertionError('settle entered')):
            rig, _trace, _a = scenario_refused_then_armed(settle=False, decode_after=2)
        self.assertEqual(rig.d.append_yield_phase(), 'resume')
        self.assertEqual((yrs(rig)['settle_waits'], yrs(rig)['settled']), (0, 0))

    def test_a_reader_fault_never_decides(self):
        with mock.patch.object(auto.AutoArmPolicy, '_yrs_bump', side_effect=RuntimeError('reader')):
            rig, trace, _a = scenario_refused_then_armed(settle=False, decode_after=2)
        _rig2, ref, _b = scenario_refused_then_armed(settle=False, decode_after=2)
        self.assertEqual(trace.result(), ref.result())
        self.assertGreater(rig.p.yrs_counts['errors'], 0)


class Settle(unittest.TestCase):
    """(2) FLASH_S5_YIELD_REENTRY_SETTLE=1."""

    def test_waits_unpaused_holding_then_the_first_barrier_passes(self):
        rig = Rig(cached={})
        rig.p.yield_settle = True
        append = window(rig)
        other = rig.add_held(ay.OTHER_UID, 6)
        handback_scheduled(rig, append)
        handback_landed(rig, append)
        for _ in range(3):
            ay.policy_ticks(rig, 1)
            self.assertEqual(rig.d.append_yield_phase(), 'handback')
            self.assertEqual(rig.p.phase, 'native')
            self.assertEqual(rig.s.pause_state, ay.Pause.UNPAUSED)
            self.assertFalse(rig.core.paused)
            # the FULL window's holds stay as today: the bulk and every other stock prefill held
            self.assertTrue(sh.defer_stock_positive_frontier(rig.s, rig.bulk))
            self.assertTrue(sh.defer_stock_prefix_hit(rig.s, NS(blocks=()), request=other))
            self.assertIsNone(ay.step_admit(rig))
        decode_scheduled(rig, append)
        ay.to_resume(rig)
        block = yrs(rig)
        self.assertEqual(block['settle_on'], 1)
        self.assertEqual((block['settle_waits'], block['settled'], block['settle_bound']), (1, 1, 0))
        self.assertEqual(block['settle_ticks'], 3)
        self.assertEqual(block['refused'], {})
        self.assertEqual((block['barrier_pass'], block['barrier_first_try']), (1, 1))
        self.assertEqual(block['retries_until_armed'], {'0': 1})
        self.assertEqual(rig.p.plan['start'], rig.bulk.num_computed_tokens)
        self.assertEqual(rig.d.admission_intent()['request_uid'], ay.BULK_UID)
        self.assertEqual(rig.d.append_yield_counters['handbacks_served'], 1)
        txn = rig.admit()
        self.assertEqual(txn.request_uid, ay.BULK_UID)

    def test_already_decoding_does_not_wait(self):
        rig = Rig(cached={})
        rig.p.yield_settle = True
        append = window(rig)
        handback_scheduled(rig, append)
        handback_landed(rig, append)
        decode_scheduled(rig, append)
        ay.to_resume(rig)
        block = yrs(rig)
        self.assertEqual((block['settle_waits'], block['settle_skipped']), (0, {'already-decoding': 1}))
        self.assertEqual(block['barrier_first_try'], 1)

    def test_bound_then_todays_path_and_the_retry_bound_unchanged(self):
        rig, on, _a = scenario_refused_then_armed(settle=True, decode_after=99, retries=2)
        block = yrs(rig)
        self.assertEqual((block['settle_waits'], block['settle_bound'], block['settled']), (1, 1, 0))
        self.assertEqual(block['settle_ticks'], auto.O3_TAIL_TICKS)
        # past the bound: today's refusals under the same bound, the same end (stock), the same journal from there
        self.assertEqual(block['refused'], {'barrier:prefill-requests': 3})
        self.assertEqual(block['exceeded'], 1)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'reentry-refused')
        _rig2, off, _b = scenario_refused_then_armed(settle=False, decode_after=99, retries=2)
        self.assertEqual(on.result()['journal'], off.result()['journal'])
        self.assertEqual(on.result()['controls'], off.result()['controls'])
        # one counter differs by design: append_yield_note_handback counts a handback 'served' when the append's first
        # token is sampled at the re-entry's start; the settle starts the re-entry after it (off: still in flight)
        on_c, off_c = dict(on.result()['counters']), dict(off.result()['counters'])
        self.assertEqual((on_c.pop('handbacks_served'), off_c.pop('handbacks_served')), (1, 0))
        self.assertEqual(on_c, off_c)

    def test_append_leaving_during_the_settle_reenters(self):
        rig = Rig(cached={})
        rig.p.yield_settle = True
        append = window(rig)
        handback_scheduled(rig, append)
        handback_landed(rig, append)
        ay.policy_ticks(rig, 2)
        self.assertEqual(rig.d.append_yield_phase(), 'handback')
        rig.stock_finish(append)
        ay.to_resume(rig)
        block = yrs(rig)
        self.assertEqual(block['settle_skipped'], {'append-left': 1})
        self.assertEqual(block['settle_waits'], 1)
        self.assertEqual(rig.d.admission_intent()['request_uid'], ay.BULK_UID)

    def test_the_expiry_path_never_settles(self):
        rig = Rig(cached={}, handback_steps=3)
        rig.p.yield_settle = True
        window(rig)
        ay.policy_ticks(rig, 3)
        self.assertEqual(rig.d.append_yield_counters['handback_expired'], 1)
        self.assertEqual(rig.d.append_yield_phase(), 'reenter')
        self.assertEqual(yrs(rig)['settle_waits'], 0)

    def test_the_off_and_on_cycles_end_in_the_same_state(self):
        ends = {}
        for settle in (False, True):
            rig, _trace, _a = scenario_refused_then_armed(settle=settle, decode_after=2)
            txn = rig.admit()
            rig.drive(lambda: rig.d.tx is not None)
            ends[settle] = dict(txn=(txn.request_uid, txn.chunk_start, txn.chunk_end), plan=dict(rig.p.plan),
                                cohort=rig.p.cohort, generation=rig.d.route_generation,
                                end=[e['reason'] for e in rig.journal('append_yield_end')])
        self.assertEqual(ends[False], ends[True])

    def test_a_settle_fault_takes_todays_path(self):
        rig = Rig(cached={})
        rig.p.yield_settle = True
        append = window(rig)
        handback_scheduled(rig, append)
        handback_landed(rig, append)
        with mock.patch.object(auto.AutoArmPolicy, '_handback_scheduled', side_effect=RuntimeError('reader')):
            ay.policy_ticks(rig, 1)
        self.assertEqual(rig.d.append_yield_phase(), 'reenter')
        self.assertGreater(yrs(rig)['errors'], 0)


class SettleStaged(ay.StagedDriver, Settle):
    """Settle on StagedSegmentDriver (the box driver)."""


class DiagnosticStaged(ay.StagedDriver, Diagnostic):
    """Diagnostic on StagedSegmentDriver (the box driver)."""


class Mutants(unittest.TestCase):
    def fails(self, check, *mutation):
        with ay.mutant(*mutation), self.assertRaises(ay.MUTANT_FAILURES + (StopIteration,)):
            check()

    def test_settle_without_its_bound(self):
        def check():
            rig, _t, _a = scenario_refused_then_armed(settle=True, decode_after=999, retries=2)
            ends = rig.journal('append_yield_end')
            self.assertTrue(ends, 'the settle never ended')
            self.assertEqual(ends[-1]['reason'], 'reentry-refused')
        check()
        self.fails(check, auto.AutoArmPolicy, '_yrs_settle_waits',
                   ("            if self._yrs_ticks > O3_TAIL_TICKS:\n", "            if False:\n"))

    def test_settle_that_pauses(self):
        check = Settle('test_waits_unpaused_holding_then_the_first_barrier_passes').test_waits_unpaused_holding_then_the_first_barrier_passes
        check()
        self.fails(check, auto.AutoArmPolicy, '_yrs_settle_waits',
                   ("            self.yrs_counts['settle_ticks'] += 1\n",
                    "            self.yrs_counts['settle_ticks'] += 1\n"
                    "            self._pause(type(self.old_pause).PAUSED_ALL)\n"))

    def test_settle_that_ignores_the_first_decode(self):
        check = Settle('test_waits_unpaused_holding_then_the_first_barrier_passes').test_waits_unpaused_holding_then_the_first_barrier_passes
        self.fails(check, auto.AutoArmPolicy, '_yrs_settle_waits',
                   ("            elif self._decode_scheduled(append):\n", "            elif False:\n"))

    def test_diagnostic_that_decides(self):
        check = Diagnostic('test_inert_against_the_same_run_without_it_flag_off_and_on').test_inert_against_the_same_run_without_it_flag_off_and_on
        self.fails(check, auto.AutoArmPolicy, '_yrs_refusal',
                   ("            if ends:\n                self.yrs_counts['exceeded'] += 1\n",
                    "            self.reentry_refusals = 0\n            if ends:\n                self.yrs_counts['exceeded'] += 1\n"))

    def test_proof_why_out_of_order(self):
        check = Diagnostic('test_proof_why_is_none_exactly_when_proof_holds').test_proof_why_is_none_exactly_when_proof_holds
        self.fails(check, auto.AutoArmPolicy, '_yrs_proof_why',
                   ("                elif row.get('barrier_clean') is not True:\n",
                    "                elif row.get('barrier_clean') is True:\n"))


if __name__ == '__main__':
    unittest.main()
