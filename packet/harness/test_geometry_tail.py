"""coverage-soak-fix (lead, 6 Nov 06:25): a geometry tail goes native to the last Mamba block boundary and hands stock its r rows.

The soak's four plan-empty nr turns (build-2 record, engine-stream.log 94961, 171967, 190148, 190153) had num_tokens % 4048 in
2..64: stock's split must stop at the last block boundary, so a native turn to num_tokens - 1 ends in a chunk under the staged
minimum and has no plan. CPU only: the real controller, auto-arm policy, wire records, worker bridge, coverage counter,
driver yield checks and measurement census, over stock's own _mamba_block_aligned_split (the reference text, unchanged).
"""
import copy
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'patches/s5_engine')]

import engine_driver as ed  # noqa: E402
import scheduler_hooks as sh  # noqa: E402
import wire_v1 as w  # noqa: E402
import lane_chunk  # noqa: E402
import test_coverage_residue as tcr  # noqa: E402
import test_e3_admission_native as adm  # noqa: E402

BLOCK, UNIT, CAP, FLOOR = tcr.BLOCK, tcr.UNIT, tcr.CAP, tcr.FLOOR
SOAK = (('A nr.0 append', 105296, 103488), ('B nr.1 fresh', 89067, 0),
        ('C nr.1 append', 101249, 98560), ('D nr.1 append', 109337, 106128))
HOST = NS(**tcr.split_host())


def lcp(n):
    return n - n % BLOCK


class Rule(unittest.TestCase):
    def rows(self, n, minimum=FLOOR, **host):
        return sh.geometry_tail_rows(NS(**dict(tcr.split_host(), **host)), NS(num_tokens=n), minimum)

    def test_r_in_two_to_sixty_four_is_the_tail(self):
        for r in (2, 11, 48, 64):
            self.assertEqual(self.rows(22 * BLOCK + r), r)
        for r in (0, 1, 65, 176, 4047):
            self.assertEqual(self.rows(22 * BLOCK + r), 0, r)

    def test_the_four_soak_turns_are_geometry_tails(self):
        self.assertEqual([self.rows(n) for _name, n, _s in SOAK], [48, 11, 49, 41])

    def test_under_one_block_eagle_or_a_small_minimum_is_no_tail(self):
        self.assertEqual(self.rows(40), 0)
        self.assertEqual(self.rows(BLOCK + 11, use_eagle_block_drop=True), 0)
        self.assertEqual(self.rows(BLOCK + 11, minimum=8), 0)
        self.assertEqual(sh.geometry_tail_rows(NS(), NS(num_tokens=BLOCK + 11), FLOOR), 0)

    def test_the_wire_and_the_hooks_pin_the_same_most_rows(self):
        self.assertEqual(sh.GEOMETRY_TAIL_MAX, w.GEOMETRY_TAIL_MAX)
        self.assertEqual(sh.GEOMETRY_TAIL_MAX, sh.QSA_STAGE_MIN_ROWS)


def bind(controller):
    """The real controller methods on the compose-5 policy rig's stand-in controller."""
    real = sh.PrefillAdmissionController
    for name in ('geometry_tail', 'native_end', 'handback_rows'):
        setattr(controller, name, types.MethodType(getattr(real, name), controller))


class Plan(unittest.TestCase):
    """The auto-arm's _plan over the soak's four turns, the residue on (nr): planned to the block boundary."""

    def policy(self, residue, n, start, geometry=True):
        hits = ((start, 0),) if start else ()
        s, r = tcr.Junction.policy(self, residue, start, 0, tokens=n, hits=hits)
        for name, value in tcr.split_host().items():
            if not hasattr(s.scheduler, name):
                setattr(s.scheduler, name, value)
        s.scheduler.cache_config = NS(block_size=BLOCK)
        if geometry:
            bind(s.controller)
        return s, r

    def ends(self, plan):
        position, out = plan['start'], []
        for size in plan['plan']:
            position += size
            out.append(position)
        return out

    def test_each_soak_turn_is_planned_to_its_last_block_boundary(self):
        for name, n, start in SOAK:
            with self.subTest(name):
                s, r = self.policy(1, n, start)
                self.assertTrue(s.policy.holds_from_stock(r))
                self.assertTrue(s.policy.tick(), s.policy._decline_why)
                plan = s.policy.plan
                self.assertEqual((plan['start'], plan['end']), (start, lcp(n)))
                self.assertEqual(self.ends(plan)[-1], lcp(n))
                self.assertTrue(all(size >= FLOOR for size in plan['plan']))
                # the hold's moved stop is not used: the native end is the block boundary itself
                self.assertEqual(s.policy.decline_record()['holds']['moved_stop'], 0)

    def test_off_the_residue_it_is_plan_empty_as_on_the_box(self):
        s, r = self.policy(0, 89067, 0)
        s.policy.holds_from_stock(r)
        self.assertFalse(s.policy.tick())
        self.assertEqual(s.policy._decline_why, 'plan-empty')

    def test_mutant_without_the_native_end_it_is_plan_empty(self):
        s, r = self.policy(1, 89067, 0, geometry=False)
        s.policy.holds_from_stock(r)
        self.assertFalse(s.policy.tick())
        self.assertEqual(s.policy._decline_why, 'plan-empty')

    def test_a_tail_of_sixty_five_rows_keeps_the_last_row(self):
        n = 22 * BLOCK + 65
        s, r = self.policy(1, n, 0)
        s.policy.holds_from_stock(r)
        self.assertTrue(s.policy.tick(), s.policy._decline_why)
        self.assertEqual(s.policy.plan['end'], n - 1)

    def test_sweep_every_tail_plans_and_others_are_unchanged(self):
        # every r in 0..130 at two block multiples, fresh and from a hash-boundary start: a plan, and its end
        for k in (2, 15):
            for r in range(0, 131):
                n = k * BLOCK + r
                for start in (0, (n - 2500) // UNIT * UNIT):
                    if start and n - 1 - start < FLOOR:
                        continue
                    s, q = self.policy(1, n, start)
                    s.policy.holds_from_stock(q)
                    self.assertTrue(s.policy.tick(), (k, r, start, s.policy._decline_why))
                    expected = lcp(n) if 2 <= r <= FLOOR else n - 1
                    self.assertEqual(s.policy.plan['end'], expected, (k, r, start))


def admission_rig(n, residue=1, start=0):
    request = adm.Request(tokens=n)
    request.num_prompt_tokens, request.shared_prefix_boundary, request.block_hashes = n, 0, ()
    manager = tcr.JunctionManager(start, 0)
    scheduler = adm.Scheduler(manager, [request])
    for name, value in tcr.split_host().items():
        setattr(scheduler, name, value)
    scheduler._mamba_block_aligned_split = types.MethodType(tcr.SPLIT, scheduler)
    controller = adm.make_controller(scheduler, chunk=CAP)
    controller.minimum_stage_rows, controller.native_hold_s, controller.native_residue = FLOOR, 30, residue
    return request, scheduler, manager, controller


def commit(controller, record):
    """tcr.commit, with each chunk's own state epoch on its reports and releases."""
    transaction = record.transaction
    controller.bind(adm.make_ticket(transaction, 0, 0, adm.DependencyRole.PRODUCER))
    for report in adm.reports_for(transaction, epoch=record.state_epoch):
        controller.note_report(report)
        controller.note_worker_release(adm.release_ack(transaction, report, epoch=record.state_epoch,
                                                       terminal=record.terminal))
    return controller.commit(transaction, adm.make_proof(transaction, 0, 0, adm.DependencyRole.PRODUCER,
                                                         adm.CompletionKind.GPU_EVENT))


def run_native(controller, scheduler, limit=64):
    """Admit and commit every native chunk; the records, in order."""
    records = []
    for _ in range(limit):
        record = controller.admit(scheduler)
        if record is None:
            return records
        records.append(record)
        commit(controller, record)
    raise AssertionError('native turn did not end')


class Admission(unittest.TestCase):
    N = 2 * BLOCK + 11          # r = 11, the fresh soak turn's

    def test_the_native_turn_ends_at_the_block_boundary_and_hands_stock_r_rows(self):
        request, scheduler, manager, controller = admission_rig(self.N)
        records = run_native(controller, scheduler)
        last = records[-1]
        self.assertTrue(last.terminal)
        self.assertEqual((last.transaction.chunk_end, last.native_end, last.remaining_tokens), (lcp(self.N), lcp(self.N), 11))
        self.assertEqual([r.terminal for r in records[:-1]], [False] * (len(records) - 1))
        self.assertEqual(request.num_computed_tokens, lcp(self.N))
        self.assertEqual(manager.cache_calls[-1], (request.request_id, lcp(self.N)))
        self.assertIs(scheduler.waiting.peek_request(), request)      # stock schedules the rest
        self.assertIsNone(controller.admit(scheduler))                # never native again
        rec = controller.coverage.record()
        self.assertEqual((rec['geometry_tail']['turns'], rec['geometry_tail']['max_r'], rec['native_turns']), (1, 11, 0))
        # stock runs the r rows in one step beside a decoder: all 'geometry-tail', native-served
        scheduler.requests = {request.request_id: request}
        scheduler.running = [tcr.decoder_running(), request]
        request.num_computed_tokens += 11
        controller.coverage.note_stock(scheduler, NS(num_scheduled_tokens={request.request_id: 11}))
        rec = controller.coverage.record()
        self.assertEqual(rec['rows']['live']['stock'], {'geometry-tail': 11})
        self.assertEqual((rec['errors'], rec['geometry_tail']['max_rows']), (0, 11))

    def test_off_the_residue_the_admission_has_no_plan(self):
        _request, scheduler, _manager, controller = admission_rig(self.N, residue=0)
        controller.native_hold_s = 30
        record = controller.admit(scheduler)
        self.assertIsNone(record)
        self.assertEqual(controller.last_refusal['reason'], 'no-staged-plan')
        self.assertEqual(controller.last_refusal['native_end'], self.N - 1)

    def test_an_append_from_a_cached_start_counts_its_history(self):
        start = BLOCK + UNIT * 3
        _request, scheduler, _manager, controller = admission_rig(self.N, start=start)
        records = run_native(controller, scheduler)
        self.assertEqual(records[0].transaction.chunk_start, start)
        rec = controller.coverage.record()
        self.assertEqual((rec['native_appends'], rec['append_history_tokens']), (1, start))

    def test_a_hit_at_the_block_boundary_is_lowered_by_the_floor_below_it(self):
        # residue_hit lowers against the native end (the boundary), not the last row: a hit at L - 60 is under the floor of L
        L = lcp(self.N)
        _request, scheduler, manager, controller = admission_rig(self.N, start=L)
        manager.coordinator = tcr.Coordinator(((L - 176, 0), (L - 60, 0)))
        manager.create_kv_cache_blocks = lambda found: None
        records = run_native(controller, scheduler)
        self.assertEqual((records[0].transaction.chunk_start, records[-1].transaction.chunk_end), (L - 176, L))
        self.assertEqual(manager.coordinator.asked, [L - FLOOR])
        self.assertEqual(controller.residue_totals['lowered'], 1)

    def test_a_fresh_turn_counts_no_history(self):
        _request, scheduler, _manager, controller = admission_rig(self.N)
        run_native(controller, scheduler)
        rec = controller.coverage.record()
        self.assertEqual((rec['native_appends'], rec['append_history_tokens']), (0, 0))

    def test_a_one_row_turn_is_unchanged(self):
        n = 2 * BLOCK + 300
        request, scheduler, _manager, controller = admission_rig(n)
        records = run_native(controller, scheduler)
        self.assertEqual((records[-1].native_end, records[-1].remaining_tokens), (n - 1, 1))
        rec = controller.coverage.record()
        self.assertEqual((rec['native_turns'], rec['geometry_tail']['turns']), (1, 0))


class Counter(unittest.TestCase):
    def setUp(self):
        self.c = sh.PrefillCoverage()

    def tail_turn(self, r=11, n=2 * BLOCK + 11):
        p = tcr.running('p', n, n - r)
        self.c.note_handback(p, r, r)
        return p

    def stock(self, p, rows):
        d = tcr.running('d', 100, 140, tokens=141)
        scheduler, out = tcr.step((d, 1, True), (p, rows, True))
        self.c.note_stock(scheduler, out)
        return self.c.record()

    def test_the_r_rows_are_geometry_tail(self):
        rec = self.stock(self.tail_turn(), 11)
        self.assertEqual(rec['rows']['live']['stock'], {'geometry-tail': 11})
        self.assertEqual(rec['requests']['live']['stock'], {'geometry-tail': 1})
        self.assertEqual(rec['geometry_tail'], dict(turns=1, max_r=11, max_rows=11))
        self.assertEqual((rec['native_turns'], rec['handback_max_rows'], rec['errors']), (0, 0, 0))

    def test_in_two_steps_still_geometry_tail(self):
        p = self.tail_turn()
        self.stock(p, 5)
        rec = self.stock(p, 6)
        self.assertEqual(rec['rows']['live']['stock'], {'geometry-tail': 11})

    def test_a_row_past_r_is_handback_excess(self):
        p = self.tail_turn()
        self.stock(p, 11)
        p.num_computed_tokens -= 2             # a preemption's replay of the last two rows
        rec = self.stock(p, 2)
        self.assertEqual(rec['rows']['live']['stock'], {'geometry-tail': 11, 'handback-excess': 2})
        self.assertEqual(rec['geometry_tail']['max_rows'], 13)

    def test_rows_not_equal_to_the_geometry_are_an_error_and_excess(self):
        p = tcr.running('p', 2 * BLOCK + 11, 2 * BLOCK)
        self.c.note_handback(p, 11, 0)        # planned a tail, stock's geometry says none
        rec = self.stock(p, 11)
        self.assertEqual(rec['rows']['live']['stock'], {'handback': 1, 'handback-excess': 10})
        self.assertEqual(rec['errors'], 1)
        c = sh.PrefillCoverage()
        c.note_handback(p, 11, 12)
        self.assertEqual((c.record()['errors'], c.record()['geometry_tail']['turns']), (1, 0))
        c.note_handback(p, 70, 70)
        self.assertEqual((c.record()['errors'], c.record()['geometry_tail']['turns']), (2, 0))

    def test_the_next_turn_of_the_request_starts_its_own_class(self):
        p = self.tail_turn()
        self.stock(p, 11)
        p.num_computed_tokens -= 1
        self.c.note_handback(p)                  # a later native turn of it ends in a one-row handback
        rec = self.stock(p, 1)
        self.assertEqual(rec['rows']['live']['stock'], {'geometry-tail': 11, 'handback': 1})
        self.assertEqual((rec['native_turns'], rec['geometry_tail']['turns']), (1, 1))

    def test_the_tail_log_names_each_step(self):
        lines = []
        logger = NS(info=lambda fmt, *a: lines.append(fmt % a))
        original = ed._progress_logger
        ed._progress_logger = lambda: logger
        try:
            self.stock(self.tail_turn(), 11)
        finally:
            ed._progress_logger = original
        self.assertEqual(lines, ['flash-s5 geometry-tail rows=11 r=11 num_tokens=%d live=1' % (2 * BLOCK + 11)])


class Wire(unittest.TestCase):
    def make(self, prompt_end, chunk_end, native_end=0):
        base = adm.Request(tokens=prompt_end)
        scheduler = adm.Scheduler(adm.Manager(), [base])
        controller = adm.make_controller(scheduler, chunk=4)
        record0 = controller.admit(scheduler)
        tx = record0.transaction
        tx = type(tx)(**dict(((f, getattr(tx, f)) for f in tx.__dataclass_fields__), chunk_end=chunk_end))
        fields = {f: getattr(record0, f) for f in record0.__dataclass_fields__}
        fields.update(transaction=tx, prompt_end=prompt_end, native_end=native_end,
                      input_ids=tuple(range(chunk_end - tx.chunk_start)))
        return w.AdmissionRecord(**fields)

    def test_default_is_the_one_row_handback(self):
        rec = self.make(12, 11)
        self.assertEqual((rec.native_end, rec.remaining_tokens, rec.terminal), (11, 1, True))
        self.assertFalse(self.make(12, 4).terminal)

    def test_a_geometry_tail_end_is_terminal_with_r_left(self):
        rec = self.make(80, 20, native_end=20)
        self.assertEqual((rec.remaining_tokens, rec.terminal), (60, True))
        self.assertFalse(self.make(80, 4, native_end=20).terminal)
        self.assertIn('native_end=20', repr(rec))

    def test_past_the_native_end_or_more_than_the_most_rows_refused(self):
        with self.assertRaises(w.E3ContractError):
            self.make(80, 30, native_end=20)
        with self.assertRaises(w.E3ContractError):
            self.make(100, 20, native_end=20)     # 80 rows left
        with self.assertRaises(w.E3ContractError):
            self.make(12, 4, native_end=12)       # the last row is never native


class Worker(unittest.TestCase):
    def chunk(self, native_end):
        tx = adm.Request  # noqa: F841 (shape only)
        base = Wire.make(Wire(), 80, 20, native_end=native_end)
        return lane_chunk.ChunkAdmission(transaction=base.transaction, prompt_end=80, input_ids=base.input_ids,
                                         ngram_context=(), block_tables=base.block_tables, native_end=native_end)

    def test_the_lane_admission_carries_the_native_end(self):
        import worker_bridge as wb
        base = Wire.make(Wire(), 80, 20, native_end=20)
        chunk = wb.admission_to_chunk(base)
        self.assertEqual(chunk.native_end, 20)
        self.assertEqual(lane_chunk.ChunkAdmission(transaction=Wire.make(Wire(), 12, 11).transaction, prompt_end=12,
                                                   input_ids=tuple(range(11 - 8)) if False else Wire.make(Wire(), 12, 11).input_ids,
                                                   ngram_context=(), block_tables=base.block_tables).native_end, 11)

    def test_the_lane_refuses_a_chunk_past_the_native_end(self):
        base = Wire.make(Wire(), 80, 20, native_end=20)
        with self.assertRaises(lane_chunk.LaneChunkError):
            lane_chunk.ChunkAdmission(transaction=base.transaction, prompt_end=80, input_ids=base.input_ids,
                                      ngram_context=(), block_tables=base.block_tables, native_end=19)


class Yield(unittest.TestCase):
    """The append yield's FULL window lets the append's r rows through and never skips them."""

    def driver(self, n, computed, rows):
        append = NS(request_id='a', num_tokens=n, num_computed_tokens=computed, num_output_tokens=0)
        d = NS(_yield=dict(phase='handback', append=append), scheduler=NS(requests={'a': append}),
               controller=NS(handback_rows=lambda s, r: rows), append_yield_counters=dict(handback_skipped=0),
               _yield_journal=lambda *a, **k: None, route_generation=1)
        d._handback_rows = types.MethodType(ed.NativeSegmentDriver._handback_rows, d)
        return d, append

    def test_r_rows_left_are_the_handback(self):
        d, a = self.driver(100, 89, 11)
        self.assertFalse(ed.NativeSegmentDriver.append_yield_holds(d, a, 'positive-frontier'))
        self.assertFalse(ed.NativeSegmentDriver.append_yield_handback_skip(d))

    def test_more_than_r_left_is_held_and_skipped(self):
        d, a = self.driver(100, 88, 11)
        self.assertTrue(ed.NativeSegmentDriver.append_yield_holds(d, a, 'positive-frontier'))
        self.assertTrue(ed.NativeSegmentDriver.append_yield_handback_skip(d))

    def test_without_the_controllers_rows_it_is_one(self):
        d, a = self.driver(100, 99, 11)
        d.controller = NS()
        self.assertFalse(ed.NativeSegmentDriver.append_yield_holds(d, a, 'positive-frontier'))
        a.num_computed_tokens = 89
        self.assertTrue(ed.NativeSegmentDriver.append_yield_holds(d, a, 'positive-frontier'))


class Census(unittest.TestCase):
    """The measurement census accepts a terminal chunk at a geometry tail's native end, and only there."""

    def journal(self):
        import test_e3_native_measurement as tm
        stack = tm.make_native_stack()
        return tm, stack, tm.run_journal(stack)

    def evidence(self, tm, stack, entries):
        return tm.classify(stack, copy.deepcopy(entries))[1]

    def shifted(self, entries, extra, native_end_known=True):
        """The same journal with the prompt ``extra`` rows longer: its terminal chunk leaves 1 + extra rows."""
        out = []
        for entry in copy.deepcopy(entries):
            if entry.get('kind') == 'chunk_open' and type(entry.get('prompt_end')) is int:
                entry['native_end'] = entry['prompt_end'] - 1 if native_end_known else None
                entry['prompt_end'] += extra
                entry['remaining_tokens'] = entry['prompt_end'] - entry['chunk_end']
                if not native_end_known:
                    entry.pop('native_end')
            frontier = entry.get('frontier') if entry.get('kind') == 'committed_frontier' else None
            if type(frontier) is dict and type(frontier.get('remaining')) is int:
                frontier['remaining'] += extra
            out.append(entry)
        return out

    def test_a_geometry_tail_terminal_is_accepted_as_the_one_row_is(self):
        tm, stack, entries = self.journal()
        base = self.evidence(tm, stack, entries)
        tail = self.evidence(tm, stack, self.shifted(entries, 10))
        self.assertEqual(tail['status'], base['status'])
        self.assertEqual(sorted(tail['reasons']), sorted(base['reasons']))

    def test_mutant_without_the_native_end_it_is_refused(self):
        tm, stack, entries = self.journal()
        tail = self.evidence(tm, stack, self.shifted(entries, 10, native_end_known=False))
        self.assertNotEqual(tail['status'], self.evidence(tm, stack, entries)['status'])


class RealStack(unittest.TestCase):
    """The real driver, wire and four real worker bridges (test_e3_native_storage_replay's stack) over a turn whose native end
    leaves stock 2 rows: the chunk ending there is terminal on every side, every worker closes the request row, the census
    accepts the request, and stock takes the 2 rows."""

    def run_stack(self, tail):
        from unittest import mock
        import test_e3_native_measurement as tm
        with mock.patch.object(sh.PrefillAdmissionController, 'geometry_tail', lambda self, s, r: tail):
            stack = tm.make_native_stack()
            entries = tm.run_journal(stack)
            census, evidence = tm.classify(stack, entries)
            stack.again = stack.controller.admit(stack.scheduler)      # the next pass, under the same tail
        return stack, entries, evidence

    def opens(self, entries):
        return [(e['chunk_start'], e['chunk_end'], e['terminal'], e.get('native_end'))
                for e in entries if e.get('kind') == 'chunk_open']

    def test_a_two_row_tail_ends_native_at_ten_of_twelve(self):
        stack, entries, evidence = self.run_stack(2)
        self.assertEqual(self.opens(entries), [(0, 4, False, 10), (4, 8, False, 10), (8, 10, True, 10)])
        self.assertEqual(evidence['status'], 'COMPLETE', evidence['reasons'])
        request = stack.scheduler.waiting.items[0] if stack.scheduler.waiting.items else None
        self.assertEqual(stack.controller.coverage.record()['geometry_tail']['turns'], 0,
                         'the stack has no block size: the class is stock\'s geometry, so this turn counts as an error, not a class')
        self.assertEqual(stack.controller.coverage.record()['errors'], 1)
        for rank in stack.driver.ranks:
            bridge = stack.worker(rank)._flash_s5_bridge
            ack = bridge._closed                                            # the last release ack, kept after the reset
            self.assertTrue(ack.request_closed, rank)                       # the terminal release closed the row at 10
            self.assertEqual(ack.transaction.chunk_end, 10)
        # stock's: the request waits at 10 of 12, and the admission never takes it natively again
        self.assertEqual((request.num_computed_tokens, request.num_tokens), (10, 12))
        self.assertIsNone(stack.again)

    def test_the_one_row_turn_is_as_before(self):
        _stack, entries, evidence = self.run_stack(0)
        self.assertEqual(self.opens(entries), [(0, 4, False, 11), (4, 8, False, 11), (8, 11, True, 11)])
        self.assertEqual(evidence['status'], 'COMPLETE', evidence['reasons'])


if __name__ == '__main__':
    unittest.main()
