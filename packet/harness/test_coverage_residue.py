"""coverage-residue: the hold build's stock residue goes native, and every prefill row is counted where it ran.

CPU only. The real PrefillCoverage over stand-in schedule() outputs and over the real admission's commit and cancel; the real
residue_hit; the real AutoArmPolicy (compose-5's stand-in scheduler); and stock's own _mamba_block_aligned_split, its text
taken from the reference scheduler and run unchanged, so the junction stop is stock's, not a copy of it.
"""
import ast
import sys
import textwrap
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'patches/s5_engine')]

import native_auto_arm as auto  # noqa: E402
import scheduler_hooks as sh  # noqa: E402
import test_compose5_declines as rig  # noqa: E402
import test_e3_admission_native as adm  # noqa: E402

REFERENCE = HERE.parent / 'reference/vllm-src/v1/core/sched/scheduler.py'
# Production geometry (e3_native_staged_plan_cpu_probe): Mamba block 4048, hash unit 176, chunk cap 1024, staged minimum 64.
BLOCK, UNIT, CAP, FLOOR = 4048, 176, 1024, 64


def stock_split():
    """Scheduler._mamba_block_aligned_split, the reference text unchanged; prefill checkpoints off (the box's config)."""
    tree = ast.parse(REFERENCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_mamba_block_aligned_split')
    source = textwrap.dedent(ast.get_source_segment(REFERENCE.read_text(), fn))
    namespace = dict(Request=object, get_mamba_prefill_checkpoint_position=lambda *a, **k: 0,
                     is_mamba_prefill_checkpoint_valid=lambda **k: False)
    exec(compile(source, str(REFERENCE), 'exec'), namespace)
    return namespace['_mamba_block_aligned_split']


SPLIT = stock_split()


def split_host(**over):
    """The scheduler fields stock's split reads, at production values."""
    fields = dict(cache_config=NS(block_size=BLOCK), hash_block_size=UNIT, use_eagle_block_drop=False,
                  mamba_has_prefill_checkpoint_blocks=False, mamba_prefill_checkpoint_alignment=BLOCK,
                  mamba_partial_cache_hit=True, mamba_fine_grained_prefix_cache=True, max_num_scheduled_tokens=CAP,
                  scheduler_config=NS(long_prefill_token_threshold=0))
    fields.update(over)
    return fields


# -- the counter ---------------------------------------------------------------------------------------------------------
def running(uid, prompt, computed, tokens=None):
    """A request as after_schedule sees it: num_computed_tokens already advanced by this step's tokens."""
    return rig.Ref(request_id=uid, num_prompt_tokens=prompt, num_tokens=prompt if tokens is None else tokens,
              num_computed_tokens=computed)


def step(*entries, policy=None):
    """entries: (request, scheduled tokens, in running). Advances num_computed_tokens as _update_after_schedule does."""
    scheduled, requests, run = {}, {}, []
    for request, n, is_running in entries:
        requests[request.request_id] = request
        if n:
            scheduled[request.request_id] = n
            request.num_computed_tokens += n
        if is_running:
            run.append(request)
    scheduler = NS(requests=requests, running=run, _flash_s5_driver=NS(auto_arm=policy))
    return scheduler, NS(num_scheduled_tokens=scheduled)


class Counter(unittest.TestCase):
    def setUp(self):
        self.c = sh.PrefillCoverage()

    def test_a_stock_prefill_chunk_beside_a_decoder_is_live_stock(self):
        d = running('d', 100, 140, tokens=141)              # decoding: 41 outputs, one row this step
        p = running('p', 5000, 0)
        policy = NS(stock_site=lambda r: 'expired')
        scheduler, out = step((d, 1, True), (p, 1024, True), policy=policy)
        self.c.note_stock(scheduler, out)
        rec = self.c.record()
        self.assertEqual(rec['rows']['live']['stock'], {'expired': 1024})
        self.assertEqual(rec['requests']['live']['stock'], {'expired': 1})
        self.assertEqual(rec['rows']['idle']['stock'], {})
        self.assertEqual((rec['errors'], rec['steps']), (0, 1))

    def test_alone_it_is_idle(self):
        p = running('p', 5000, 0)
        scheduler, out = step((p, 1024, True))
        self.c.note_stock(scheduler, out)
        self.assertEqual(self.c.record()['rows'], {'live': {'native': {}, 'stock': {}}, 'idle': {'native': {}, 'stock': {'no-policy': 1024}}})

    def test_two_stock_prefills_are_not_each_others_decoders(self):
        a, b = running('a', 5000, 0), running('b', 300, 0)
        scheduler, out = step((a, 700, True), (b, 300, True))
        self.c.note_stock(scheduler, out)
        self.assertEqual(self.c.record()['rows']['idle']['stock'], {'no-policy': 1000})

    def test_decode_rows_are_not_prefill(self):
        d = running('d', 100, 140, tokens=141)
        scheduler, out = step((d, 1, True))
        self.c.note_stock(scheduler, out)
        rec = self.c.record()
        self.assertEqual((rec['rows']['live']['stock'], rec['rows']['idle']['stock'], rec['steps']), ({}, {}, 1))

    def test_the_last_prompt_row_counts_and_the_step_after_does_not(self):
        p = running('p', 1000, 990)
        scheduler, out = step((p, 10, True))                # rows 990..999: prefill, it ends the prompt
        self.c.note_stock(scheduler, out)
        p.num_tokens = 1001                                 # its first output appended
        scheduler, out = step((p, 1, True))
        self.c.note_stock(scheduler, out)
        self.assertEqual(self.c.record()['rows']['idle']['stock'], {'no-policy': 10})

    def test_a_request_that_finished_prefill_this_step_is_not_yet_a_decoder(self):
        p = running('p', 1000, 990)
        q = running('q', 4000, 0)
        scheduler, out = step((p, 10, True), (q, 512, True))
        self.c.note_stock(scheduler, out)
        self.assertEqual(self.c.record()['rows']['idle']['stock'], {'no-policy': 522})

    def test_a_running_decoder_not_scheduled_this_step_is_live(self):
        d = running('d', 100, 140, tokens=141)
        p = running('p', 5000, 0)
        scheduler, out = step((d, 0, True), (p, 64, True))
        self.c.note_stock(scheduler, out)
        self.assertEqual(self.c.record()['rows']['live']['stock'], {'no-policy': 64})

    def test_a_resumed_request_replays_its_outputs_as_prefill(self):
        r = running('r', 1000, 0, tokens=1300)              # preempted after 300 outputs: 1299 rows of prefill
        d = running('d', 100, 140, tokens=141)
        scheduler, out = step((d, 1, True), (r, 1024, True))
        self.c.note_stock(scheduler, out)
        scheduler, out = step((d, 1, True), (r, 1024, True))
        self.c.note_stock(scheduler, out)
        self.assertEqual(self.c.record()['rows']['live']['stock'], {'no-policy': 1299})
        self.assertEqual(self.c.record()['requests']['live']['stock'], {'no-policy': 1})

    def test_the_site_is_fixed_at_the_first_stock_row(self):
        d = running('d', 100, 140, tokens=141)
        p = running('p', 5000, 0)
        sites = iter(('expired', 'other'))
        policy = NS(stock_site=lambda r: next(sites))
        for _ in range(2):
            scheduler, out = step((d, 1, True), (p, 1024, True), policy=policy)
            self.c.note_stock(scheduler, out)
        self.assertEqual(self.c.record()['rows']['live']['stock'], {'expired': 2048})

    def test_the_handed_back_row_is_named_handback(self):
        d = running('d', 100, 140, tokens=141)
        p = running('p', 5000, 4999)
        self.c.note_handback(p)
        scheduler, out = step((d, 1, True), (p, 1, True), policy=NS(stock_site=lambda r: 'wrong'))
        self.c.note_stock(scheduler, out)
        self.assertEqual(self.c.record()['rows']['live']['stock'], {'handback': 1})

    def test_handback_is_bound_to_the_request_object_not_its_uid(self):
        d = running('d', 100, 140, tokens=141)
        self.c.note_handback(running('p', 5000, 4999))
        p = running('p', 5000, 0)                           # same uid, another request: not a handback
        scheduler, out = step((d, 1, True), (p, 64, True), policy=NS(stock_site=lambda r: 'no-seam'))
        self.c.note_stock(scheduler, out)
        self.assertEqual(self.c.record()['rows']['live']['stock'], {'no-seam': 64})

    def test_a_count_never_raises_it_counts_an_error(self):
        self.c.note_stock(NS(requests=None, running=()), NS(num_scheduled_tokens={'x': 3}))
        self.c.note_native_commit(NS(request_uid='x', chunk_start=0, chunk_end=1))      # never admitted
        self.assertEqual(self.c.record()['errors'], 2)

    def test_the_policys_site_error_is_a_site_not_a_raise(self):
        d, p = running('d', 100, 140, tokens=141), running('p', 5000, 0)
        def boom(r):
            raise RuntimeError('x')
        scheduler, out = step((d, 1, True), (p, 64, True), policy=NS(stock_site=boom))
        self.c.note_stock(scheduler, out)
        self.assertEqual(self.c.record()['errors'], 1)

    def test_after_schedule_counts_before_anything_else(self):
        d, p = running('d', 100, 140, tokens=141), running('p', 5000, 0)
        scheduler, out = step((d, 1, True), (p, 256, True))
        scheduler._flash_s5_admission = NS(coverage=self.c)
        sh.after_schedule(scheduler, out)
        self.assertEqual(self.c.record()['rows']['live']['stock'], {'no-policy': 256})

    def test_the_tables_are_bounded(self):
        for i in range(sh.MAX_COVERAGE_REQUESTS + 5):
            self.c.note_handback(running('h%d' % i, 10, 9))
        self.assertEqual(len(self.c._handback), sh.MAX_COVERAGE_REQUESTS)


# -- the real admission: native rows at commit, none at cancel ----------------------------------------------------------
def decoder_running(uid='dec'):
    return NS(request_id=uid, num_prompt_tokens=10, num_tokens=11, num_computed_tokens=10, is_prefill_chunk=False)


def commit(controller, record, terminal=False):
    transaction = record.transaction
    controller.bind(adm.make_ticket(transaction, 0, 0, adm.DependencyRole.PRODUCER))
    for report in adm.reports_for(transaction):
        controller.note_report(report)
        controller.note_worker_release(adm.release_ack(transaction, report, epoch=record.state_epoch, terminal=terminal))
    return controller.commit(transaction, adm.make_proof(transaction, 0, 0, adm.DependencyRole.PRODUCER,
                                                         adm.CompletionKind.GPU_EVENT))


class NativeRows(unittest.TestCase):
    def admit(self, tokens, running=()):
        request = adm.Request(tokens=tokens)
        scheduler = adm.Scheduler(adm.Manager(), [request])
        scheduler.running = list(running)
        controller = adm.make_controller(scheduler, chunk=8)
        return request, scheduler, controller, controller.admit(scheduler)

    def test_a_committed_chunk_counts_native_live_and_its_last_row_is_the_handback(self):
        request, scheduler, controller, record = self.admit(5, [decoder_running()])
        self.assertEqual(record.remaining_tokens, 1)
        self.assertEqual(controller.coverage.record()['open_native'], 1)
        commit(controller, record, terminal=True)
        rec = controller.coverage.record()
        self.assertEqual(rec['rows']['live']['native'], {'plan': 4})
        self.assertEqual(rec['open_native'], 0)
        # stock runs the last row: named, and counted as stock
        request.num_prompt_tokens = 5
        request.num_computed_tokens = 5
        scheduler.requests = {request.request_id: request}
        scheduler.running = [decoder_running(), request]
        controller.coverage.note_stock(scheduler, NS(num_scheduled_tokens={request.request_id: 1}))
        rec = controller.coverage.record()
        self.assertEqual(rec['rows']['live']['stock'], {'handback': 1})
        self.assertEqual(rec['errors'], 0)

    def test_with_no_decoder_it_is_idle_native(self):
        _r, _s, controller, record = self.admit(5)
        commit(controller, record, terminal=True)
        self.assertEqual(controller.coverage.record()['rows']['idle']['native'], {'plan': 4})

    def test_a_cancelled_chunk_counts_nothing(self):
        _r, _s, controller, record = self.admit(5, [decoder_running()])
        controller.cancel(record.transaction, never_submitted=True)
        rec = controller.coverage.record()
        self.assertEqual((rec['rows']['live']['native'], rec['open_native'], rec['errors']), ({}, 0, 0))


# -- residue_hit ----------------------------------------------------------------------------------------------------------
class Coordinator:
    def __init__(self, hits):
        self.hits, self.asked = hits, []

    def find_longest_cache_hit(self, hashes, limit):
        self.asked.append(limit)
        hit, uncached = max(((h, u) for h, u in self.hits if h <= limit), default=(0, 0))
        return ('found', hit), hit, uncached


def residue_manager(start, junction, hits=()):
    coordinator = Coordinator(hits)
    return NS(get_computed_blocks=lambda r: (NS(blocks=('stock', start)), start, junction), coordinator=coordinator,
              create_kv_cache_blocks=lambda found: NS(blocks=found))


class ResidueHit(unittest.TestCase):
    def hit(self, tokens, start, junction=0, hits=()):
        manager = residue_manager(start, junction, hits)
        return sh.residue_hit(NS(kv_cache_manager=manager), NS(num_tokens=tokens, block_hashes=()), FLOOR), manager

    def test_no_hit_and_a_long_tail_are_stocks_own_lookup(self):
        (blocks, start, junction, lowered), m = self.hit(20000, 0)
        self.assertEqual((start, lowered, m.coordinator.asked), (0, False, []))
        (blocks, start, junction, lowered), m = self.hit(20000, UNIT * 100, junction=UNIT * 103)
        self.assertEqual((start, junction, lowered, blocks.blocks), (UNIT * 100, UNIT * 103, False, ('stock', UNIT * 100)))

    def test_a_hit_that_leaves_under_the_floor_is_lowered(self):
        tokens = UNIT * 100 + 10                            # stock's hit: the prompt's last hash unit, 9 rows left
        (blocks, start, junction, lowered), m = self.hit(tokens, UNIT * 100, hits=((UNIT * 99, 0), (UNIT * 100, 0)))
        self.assertEqual(m.coordinator.asked, [tokens - 1 - FLOOR])
        self.assertEqual((start, junction, lowered, blocks.blocks), (UNIT * 99, 0, True, ('found', UNIT * 99)))
        self.assertGreaterEqual(tokens - 1 - start, FLOOR)

    def test_a_lowered_hit_keeps_its_own_junction(self):
        tokens = UNIT * 100 + 10
        (blocks, start, junction, lowered), _m = self.hit(tokens, UNIT * 100, hits=((UNIT * 96, UNIT * 2),))
        self.assertEqual((start, junction, lowered), (UNIT * 96, UNIT * 98, True))

    def test_nothing_lower_is_a_hit_of_zero(self):
        (blocks, start, junction, lowered), _m = self.hit(UNIT + 10, UNIT)
        self.assertEqual((start, lowered), (0, True))

    def test_a_prompt_shorter_than_the_floor_is_left_alone(self):
        (blocks, start, junction, lowered), m = self.hit(40, 32)
        self.assertEqual((start, lowered, m.coordinator.asked), (32, False, []))


# -- the junction: stock's split, in the admission and the auto-arm's plan ------------------------------------------------
class JunctionManager(adm.Manager):
    def __init__(self, cached, junction):
        super().__init__(cached=cached)
        self.junction = junction
        self.coordinator = Coordinator(())

    def get_computed_blocks(self, request):
        self.lookup_calls += 1
        return None, self.cached, self.junction


class Junction(unittest.TestCase):
    TOKENS, START, JUNCTION = 20000, BLOCK, BLOCK + UNIT * 3       # a sub-block junction inside the first chunk

    def admission(self, residue):
        request = adm.Request(tokens=self.TOKENS)
        request.num_prompt_tokens, request.shared_prefix_boundary, request.block_hashes = self.TOKENS, 0, ()
        manager = JunctionManager(self.START, self.JUNCTION)
        scheduler = adm.Scheduler(manager, [request])
        for name, value in split_host().items():
            setattr(scheduler, name, value)
        scheduler._mamba_block_aligned_split = types.MethodType(SPLIT, scheduler)
        controller = adm.make_controller(scheduler, chunk=CAP)
        controller.minimum_stage_rows, controller.native_hold_s, controller.native_residue = FLOOR, 30, residue
        return request, manager, controller, controller.admit(scheduler)

    def test_on_the_first_chunk_stops_at_the_junction_and_its_commit_publishes_there(self):
        request, manager, controller, record = self.admission(1)
        self.assertEqual(request.shared_prefix_boundary, self.JUNCTION)
        self.assertEqual((record.transaction.chunk_start, record.transaction.chunk_end), (self.START, self.JUNCTION))
        commit(controller, record)
        self.assertEqual(manager.cache_calls, [(request.request_id, self.JUNCTION)])
        self.assertEqual(controller.residue_totals, dict(junction=1, lowered=0))
        self.assertEqual(controller.coverage.record()['rows']['idle']['native'], {'junction': self.JUNCTION - self.START})

    def test_off_the_admission_is_todays_and_runs_past_it(self):
        request, manager, controller, record = self.admission(0)
        self.assertEqual(request.shared_prefix_boundary, 0)
        self.assertEqual(record.transaction.chunk_end, self.START + CAP)
        self.assertEqual(controller.residue_totals, dict(junction=0, lowered=0))

    def policy(self, residue, start, junction, tokens=20000, hits=()):
        r = rig.req(tokens=tokens, num_prompt_tokens=tokens, shared_prefix_boundary=0, block_hashes=())
        s = rig.Stage(r)
        s.scheduler.kv_cache_manager.get_computed_blocks = lambda q: (NS(blocks=()), start, junction)
        s.scheduler.kv_cache_manager.coordinator.find_longest_cache_hit = Coordinator(hits).find_longest_cache_hit
        s.scheduler.kv_cache_manager.create_kv_cache_blocks = lambda found: NS(blocks=())
        host = NS(**split_host())
        s.controller._aligned_split = lambda sched, q, limit, **kw: SPLIT(host, q, limit, **kw)
        # the hold's moved stop (a1-trunk 1d), the real controller methods, over the same geometry
        real = sh.PrefillAdmissionController
        s.scheduler.hash_block_size, s.scheduler.mamba_partial_cache_hit = UNIT, True
        s.controller._append_yield_short_tail = types.MethodType(real._append_yield_short_tail, s.controller)
        s.controller._append_yield_split_view = types.MethodType(real._append_yield_split_view, s.controller)
        s.controller.chunk_tokens, s.controller.minimum_stage_rows = CAP, FLOOR
        s.controller.native_hold_s, s.controller.native_residue = 30, residue
        s.driver.clock = lambda: 0.0
        return s, r

    def test_the_plan_takes_the_junction_and_ends_a_chunk_on_it(self):
        s, r = self.policy(1, self.START, BLOCK + UNIT * 9)
        self.assertTrue(s.policy.holds_from_stock(r))
        self.assertTrue(s.policy.tick())
        ends, position = [], s.policy.plan['start']
        for size in s.policy.plan['plan']:
            position += size
            ends.append(position)
        self.assertIn(BLOCK + UNIT * 9, ends)
        self.assertEqual(position, 20000 - 1)
        rec = s.policy.decline_record()
        self.assertEqual((rec['residue']['junction'], rec['residue']['on'], rec['holds']['selected']), (1, 1, 1))

    def test_off_plan_shared_is_declined_and_released_to_stock(self):
        s, r = self.policy(0, self.START, BLOCK + UNIT * 9)
        self.assertTrue(s.policy.holds_from_stock(r))
        self.assertFalse(s.policy.tick())
        self.assertFalse(s.policy.holds_from_stock(r))
        self.assertEqual(s.policy.stock_site(r), 'plan-shared')
        self.assertEqual(s.policy.decline_record()['residue']['on'], 0)

    def test_residue_needs_the_hold(self):
        s, r = self.policy(1, self.START, BLOCK + UNIT * 9)
        s.controller.native_hold_s = 0
        self.assertEqual(s.policy.decline_record()['residue']['on'], 0)
        self.assertFalse(s.policy.holds_from_stock(r))
        self.assertEqual(s.policy.stock_site(r), 'hold-off/none')

    def test_a_hit_in_the_last_hash_unit_is_lowered_and_planned(self):
        tokens = UNIT * 100 + 10
        s, r = self.policy(1, UNIT * 100, 0, tokens=tokens, hits=((UNIT * 96, 0), (UNIT * 100, 0)))
        self.assertTrue(s.policy.holds_from_stock(r))
        self.assertTrue(s.policy.tick())
        self.assertEqual(s.policy.plan['start'], UNIT * 96)
        self.assertEqual(s.policy.decline_record()['residue']['lowered'], 1)

    def test_off_that_hit_is_below_the_floor(self):
        tokens = UNIT * 100 + 10
        s, r = self.policy(0, UNIT * 100, 0, tokens=tokens, hits=((UNIT * 96, 0),))
        s.policy.holds_from_stock(r)
        self.assertFalse(s.policy.tick())
        self.assertEqual(s.policy._decline_why, 'plan-below-floor')


# -- refused selections: held, tried again when the cohort changes --------------------------------------------------------
def held_stage(residue):
    s = rig.Stage(rig.req())
    s.controller.native_hold_s, s.controller.native_residue = 30, residue
    s.driver.clock = lambda: 0.0
    return s


class Refused(unittest.TestCase):
    def refuse(self, s, r):
        """The selection's own refusal path: _remember under the cohort it was selected with."""
        self.assertTrue(s.policy.tick())
        s.policy._remember()
        s.policy.phase, s.policy.cooldown = 'idle', False
        s.policy.request_ref = None
        self.assertTrue(s.policy.holds_from_stock(r))       # back to waiting: held again

    def test_the_same_cohort_holds_it_with_no_retry_count(self):
        s = held_stage(1)
        r = s.queue[0]
        s.policy.holds_from_stock(r)
        self.refuse(s, r)
        for _ in range(auto.MAX_RETRIES + 3):
            self.assertFalse(s.policy.tick())
            self.assertTrue(s.policy.holds_from_stock(r))   # never released for the refusal
        self.assertEqual(s.policy.decline_record()['residue']['refused_held'], auto.MAX_RETRIES + 3)

    def test_a_changed_cohort_retries_it(self):
        s = held_stage(1)
        r = s.queue[0]
        s.policy.holds_from_stock(r)
        self.refuse(s, r)
        s.scheduler.running = [rig.decoder(), rig.decoder('d2')]
        self.assertTrue(s.policy.tick())
        rec = s.policy.decline_record()['residue']
        self.assertEqual((rec['refused_retried'], rec['refused_held']), (1, 0))

    def test_off_the_hold_builds_retry_limit_stands(self):
        s = held_stage(0)
        r = s.queue[0]
        s.policy.holds_from_stock(r)
        for _ in range(auto.MAX_RETRIES):                 # the first selection and MAX_RETRIES - 1 retries, refused
            self.refuse(s, r)
        self.assertTrue(s.policy.tick())                    # the last retry, refused too
        s.policy._remember()
        self.assertFalse(s.policy.holds_from_stock(r))      # stock takes it, by name
        self.assertEqual(s.policy.stock_site(r), 'retries-spent')


class Idle(unittest.TestCase):
    """The lead's 01:10 target: 100 % of all prefill rows, idle included. With the residue, nothing running is no reason for stock."""

    def test_with_nothing_running_it_is_held_and_selected_with_an_empty_cohort(self):
        s = held_stage(1)
        r = s.queue[0]
        s.scheduler.running = []
        self.assertTrue(s.policy.holds_from_stock(r))
        self.assertTrue(s.policy.tick())
        self.assertEqual(s.policy.cohort, ())
        self.assertEqual(s.policy.decline_record()['residue']['idle_selected'], 1)

    def test_off_nothing_running_goes_stock_and_the_cohort_is_empty_as_today(self):
        s = held_stage(0)
        r = s.queue[0]
        s.scheduler.running = []
        self.assertFalse(s.policy.holds_from_stock(r))
        self.assertEqual(s.policy.stock_site(r), 'nothing-running')
        self.assertFalse(s.policy.tick())
        self.assertEqual(s.policy._decline_why, 'cohort-empty')


def closing(residue):
    s = held_stage(residue)
    r = s.queue[0]
    s.driver.append_yield_end = lambda reason: None
    s.policy.holds_from_stock(r)
    return s, r


def close(s, r, reason, via_rollback=False):
    s.policy.request_ref = auto.weakref.ref(r)
    s.policy.old_pause, s.policy.external_pause = rig.Pause.UNPAUSED, False
    if via_rollback:
        s.policy._rollback(reason)            # no split: straight to _restore
    else:
        s.policy._restore(reason)


class Races(unittest.TestCase):
    """The hold box's 'cooldown' turns: three quick closes, then stock. A lost race is not a refusal of the request."""

    def test_a_race_close_does_not_mark_the_request_refused(self):
        for reason, via in (('selection-changed-during-output-drain', False), ('worker-barrier-cohort-refused', True), ('selection-changed-before-arm', True)):
            s, r = closing(1)
            close(s, r, reason, via)
            self.assertNotIn(r.request_id, s.policy.rejected, reason)
            self.assertEqual(s.policy.decline_record()['residue']['races'], 1)

    def test_the_rollback_reason_survives_the_return_to_full(self):
        s, r = closing(1)
        s.policy._rollback_reason = 'selection-changed-before-arm'
        close(s, r, 'returned-before-native-admission')
        self.assertNotIn(r.request_id, s.policy.rejected)

    def test_races_are_bounded_then_it_is_a_refusal(self):
        s, r = closing(1)
        for _ in range(auto.MAX_RACES):
            close(s, r, 'selection-changed-during-output-drain')
        self.assertNotIn(r.request_id, s.policy.rejected)
        close(s, r, 'selection-changed-during-output-drain')
        self.assertIn(r.request_id, s.policy.rejected)
        self.assertEqual(s.policy.decline_record()['residue']['races_spent'], 1)

    def test_a_real_refusal_still_marks_it(self):
        s, r = closing(1)
        close(s, r, 'worker-graph-support-refused', True)
        self.assertIn(r.request_id, s.policy.rejected)

    def test_off_every_close_marks_it_as_the_hold_does(self):
        s, r = closing(0)
        close(s, r, 'selection-changed-during-output-drain')
        self.assertIn(r.request_id, s.policy.rejected)


class StockSite(unittest.TestCase):
    def test_every_hold_refusal_names_itself(self):
        s = held_stage(1)
        r = s.queue[0]
        s.policy.manual_owned = True
        s.policy.holds_from_stock(r)
        self.assertEqual(s.policy.stock_site(r), 'manual-owned')
        s.policy.manual_owned = False
        s.scheduler.running = []
        self.assertTrue(s.policy.holds_from_stock(r))          # under the residue nothing running is held too (Idle)
        s.scheduler.running = [rig.decoder()]
        s.driver.clock = lambda: 1 / 0
        s.policy.holds_from_stock(r)
        self.assertEqual(s.policy.stock_site(r), 'hold-error')

    def test_expiry_names_itself(self):
        s = held_stage(1)
        r = s.queue[0]
        now = [0.0]
        s.driver.clock = lambda: now[0]
        self.assertTrue(s.policy.holds_from_stock(r))
        now[0] = 31.0
        self.assertFalse(s.policy.holds_from_stock(r))
        self.assertEqual(s.policy.stock_site(r), 'expired')

    def test_a_request_the_seam_never_saw_is_no_seam(self):
        s = held_stage(1)
        self.assertEqual(s.policy.stock_site(rig.req('other')), 'no-seam')

    def test_the_record_carries_the_counter(self):
        s = held_stage(1)
        s.controller.coverage = sh.PrefillCoverage()
        self.assertEqual(s.policy.decline_record()['coverage']['schema'], sh.COVERAGE_SCHEMA)


if __name__ == '__main__':
    unittest.main()
