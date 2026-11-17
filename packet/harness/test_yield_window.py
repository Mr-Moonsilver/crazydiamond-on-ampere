"""yield-rules Y2 (lead, 12 Nov 2026): FLASH_S5_YIELD_WINDOW, the batched yield window. CPU only.

Box evidence (V1 j2 records; builder-tail-route-20261112/STATUS-yield-rules.md): 440 of 651 junction yields are follow-ons
and 260 of them come after exactly one owner chunk: the owner re-enters, runs one chunk and yields again.

  Y2  with the flag at 1 a yield is a window. When the current append's handback was served (on FULL, or kept split by
      O3), auto-arm asks the driver for the next waiting eligible append (today's candidate: FCFS, max_new, the slot,
      decode-cover and KV-block gates per candidate) before the re-entry; a staged append makes the re-entry's arm start
      it (phase 'append') instead of the owner's resume. The owner re-enters as today when none waits, at
      YIELD_WINDOW_MAX_APPENDS (the owner's starvation bound), after an unserved handback (abort, skip, expiry) or when
      the staged append left before the arm. PER_BULK counts windows. A geometry-tail append's next append runs only
      after its FULL handback of r rows (counted: tail_full_handbacks). Counters in queue_policy.yield_window
      (windows, appends, per-window histogram, ends, in-window gap ms, owner re-entries avoided), one
      'flash-s5 n3-yield-window' receipt per append and per closed window. Flag at 0: no window call at all.

Rig: test_append_yield.Rig (real PrefillAdmissionController, segment driver, AutoArmPolicy, four bridge workers) and
test_lane_o3.yield_rig (O3 installed and selected). Bulk: 12 tokens, chunks 4/4/3; the window opens at frontier 4.
No GPU, CUDA or box claim.
"""
import inspect
import json
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
import test_e3_admission_native as adm  # noqa: E402
import test_lane_o3 as lo3  # noqa: E402
import test_tail_select as ts  # noqa: E402
import engine_driver as ed  # noqa: E402
import native_auto_arm as auto  # noqa: E402
import scheduler_hooks as sh  # noqa: E402
from e3_types import E3ContractError  # noqa: E402

NS = types.SimpleNamespace
Rig = ay.Rig
BULK_UID, APPEND_UID, OTHER_UID = ay.BULK_UID, ay.APPEND_UID, ay.OTHER_UID
THIRD_UID, FOURTH_UID = 'yw-third', 'yw-fourth'
RECEIPT = 'flash-s5 n3-yield-window '


class Lines:
    """The driver progress logger, captured (receipts are its lines)."""
    def __init__(self, tc):
        self.lines = []
        logger = NS(info=lambda fmt, *a: self.lines.append(fmt % a), warning=lambda *a, **k: None)
        patch = mock.patch.object(ed, '_progress_logger', return_value=logger)
        patch.start()
        tc.addCleanup(patch.stop)

    def receipts(self):
        return [json.loads(line[len(RECEIPT):]) for line in self.lines if line.startswith(RECEIPT)]


def window_rig(on=True, **kw):
    rig = Rig(cached={}, **kw)
    rig.c.yield_window = on
    rig.s.max_num_running_reqs = 8          # room for the window's decoders (the slot gate is per candidate)
    return rig


def served_first(rig, waiters=(OTHER_UID,), tokens=6):
    """Window opened at frontier 4 with APPEND (11 tokens, 8 cached); ``waiters`` held behind it; APPEND's natively
    computed rows committed, the route FULL and its handback served by stock."""
    append = ay.run_to_yield(rig)
    held = [rig.add_held(uid, tokens) for uid in waiters]
    ay.to_append(rig)
    ay.run_append(rig)
    ay.to_full(rig)
    rig.stock_handback(append)
    return append, held


def to_window_append(rig):
    rig.tick(lambda: rig.d.append_yield_phase() == 'append')


def counts(rig):
    w = rig.c.yield_window_counts
    return dict(w, per_window=dict(w['per_window']), ends=dict(w['ends']), gap_ms=dict(w['gap_ms']))


def kinds(rig):
    return [k[len('append_yield_'):] for k in rig.kinds() if k.startswith('append_yield_')]


class Setting(unittest.TestCase):
    def controller(self, env):
        with mock.patch.dict(os.environ, env):
            scheduler = NS(kv_cache_manager=NS(), max_num_scheduled_tokens=1024)
            return sh.PrefillAdmissionController(scheduler, leases=sh.SchedulerLeases(), frontier=adm.ah.CommittedFrontier(),
                                                 engine_epoch=7, chunk_tokens=4, ranks=(0, 1), max_history=64)

    def test_strict_boolean_default_off(self):
        env = {k: v for k, v in os.environ.items() if k != sh.YIELD_WINDOW_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(self.controller({}).yield_window)
        self.assertFalse(self.controller({sh.YIELD_WINDOW_ENV: '0'}).yield_window)
        self.assertTrue(self.controller({sh.YIELD_WINDOW_ENV: '1'}).yield_window)
        for bad in ('2', 'yes', 'true', '', ' 1'):
            with self.subTest(bad=bad), self.assertRaises(E3ContractError):
                self.controller({sh.YIELD_WINDOW_ENV: bad})

    def test_status_block_only_with_the_flag(self):
        off = self.controller({sh.YIELD_WINDOW_ENV: '0'}).native_admission_record()
        self.assertNotIn('yield_window', off.get('queue_policy') or {})
        on = self.controller({sh.YIELD_WINDOW_ENV: '1'}).native_admission_record()['queue_policy']['yield_window']
        self.assertEqual(on['max_appends'], sh.YIELD_WINDOW_MAX_APPENDS)
        self.assertEqual(sh.YIELD_WINDOW_MAX_APPENDS, 4)
        self.assertEqual((on['windows'], on['appends'], on['reentries_avoided'], on['per_window'], on['ends']),
                         (0, 0, 0, {}, {}))


class Window(unittest.TestCase):
    """The FULL-window path (O3 not installed): the state machine end to end."""

    def setUp(self):
        self.log = Lines(self)

    def test_two_appends_one_window_then_the_owner(self):
        rig = window_rig()
        d, c = rig.d, rig.c
        generation = d.route_generation
        append, (other,) = served_first(rig)
        frontier = rig.bulk.num_computed_tokens
        to_window_append(rig)
        self.assertIs(d._yield['append'], other)
        self.assertEqual(d._yield['frontier'], frontier)
        self.assertEqual(d.route_generation, generation + 2, 'FULL for the handback, then split for the next append')
        self.assertEqual(d.admission_intent()['request_uid'], BULK_UID, 'the window keeps the owner binding')
        self.assertEqual(rig.p.cohort, (APPEND_UID,))
        self.assertEqual(d._yield_counts, {(BULK_UID, d._yield['bulk_transaction'].request_generation): 1})
        self.assertNotIn('bulk_reopened', kinds(rig))
        transactions = ay.run_append(rig)
        self.assertEqual([(t.chunk_start, t.chunk_end) for t in transactions], [(0, 4), (4, 5)])
        self.assertFalse(set(rig.rows(OTHER_UID)) & set(rig.rows(BULK_UID)))
        ay.to_full(rig)
        rig.stock_handback(other)
        ay.to_resume(rig)                                   # nothing else waits: the owner re-enters
        self.assertEqual(rig.p.cohort, tuple(sorted((APPEND_UID, OTHER_UID))))
        txn = rig.admit()
        self.assertEqual((txn.request_uid, txn.chunk_start), (BULK_UID, frontier))
        rig.finish_chunk()
        ay.finish_bulk(rig)
        self.assertEqual(kinds(rig), ['selected', 'drained', 'row_closed', 'admitted', 'handback', 'full', 'window_next',
                                      'reenter', 'window_append', 'admitted', 'handback', 'full', 'reenter', 'resume',
                                      'bulk_reopened', 'end'])
        self.assertEqual(d.append_yield_counters, dict(selected=1, rows_closed=1, admitted=2, handbacks=2, reentries=2,
                                                       reopened=1, abandoned=0, handback_expired=0, reentry_refused=0,
                                                       append_refused=0, handbacks_served=2, handback_skipped=0))
        w = counts(rig)
        self.assertEqual((w['windows'], w['appends'], w['reentries_avoided'], w['tail_full_handbacks']), (1, 2, 1, 0))
        self.assertEqual((w['per_window'], w['ends']), ({'2': 1}, {'none-waiting': 1}))
        self.assertEqual(w['gap_ms']['n'], 1)
        self.assertGreaterEqual(w['gap_ms']['total'], 0.0)
        self.assertEqual(sum(w['gap_ms']['buckets'].values()), 1)
        history = d.submit_control({'action': 'status'})['append_yield']['history']
        self.assertEqual(history[-1]['window'], dict(appends=2, end='none-waiting'))
        self.assertEqual(history[-1]['reason'], 'reopened')
        block = c.native_admission_record()['queue_policy']['yield_window']
        self.assertEqual((block['windows'], block['appends']), (1, 2))
        receipts = self.log.receipts()
        self.assertEqual([(r['event'], r.get('index'), r.get('appends')) for r in receipts],
                         [('append', 1, None), ('append', 2, None), ('closed', None, 2)])
        self.assertIsNone(receipts[0]['gap_ms'])
        self.assertEqual(type(receipts[1]['gap_ms']), float)
        for line in self.log.lines:
            if line.startswith(RECEIPT):
                self.assertNotIn(BULK_UID, line)
                self.assertNotIn(OTHER_UID, line)

    def test_off_the_second_append_needs_its_own_yield(self):
        """The control: today the owner re-enters, runs one chunk and yields again for the other append."""
        rig = window_rig(on=False)
        append, (other,) = served_first(rig)
        ay.to_resume(rig)
        self.assertIsNotNone(rig.admit())
        rig.finish_chunk()
        self.assertEqual(rig.d.append_yield_phase(), 'closing')
        self.assertIs(rig.d._yield['append'], other)
        ay.to_append(rig)
        ay.run_append(rig)
        ay.to_full(rig)
        rig.stock_handback(other)
        ay.to_resume(rig)
        ay.finish_bulk(rig)
        self.assertEqual(rig.d.append_yield_counters['selected'], 2)
        self.assertEqual(rig.d.append_yield_counters['reopened'], 2)
        self.assertEqual(len(rig.journal('request_row_closed')), 2)
        self.assertNotIn('window_next', kinds(rig))
        self.assertEqual(counts(rig)['appends'], 0)
        self.assertFalse([line for line in self.log.lines if line.startswith(RECEIPT)])

    def test_the_cap_counts_windows(self):
        """PER_BULK 1: the window still takes both appends (one window); off, the second append gets no yield."""
        rig = window_rig(per_bulk=1)
        append, (other,) = served_first(rig)
        to_window_append(rig)
        ay.run_append(rig)
        ay.to_full(rig)
        rig.stock_handback(other)
        ay.to_resume(rig)
        ay.finish_bulk(rig)
        self.assertEqual(rig.d.append_yield_counters['selected'], 1)
        self.assertEqual(rig.d.append_yield_counters['admitted'], 2)
        self.assertEqual(list(rig.d._yield_counts.values()), [1])
        off = window_rig(on=False, per_bulk=1)
        served_first(off)
        ay.to_resume(off)
        ay.finish_bulk(off)
        self.assertEqual(off.d.append_yield_counters['admitted'], 1)
        self.assertIn(OTHER_UID, [r.request_id for r in off.s.skipped_waiting.items], 'off: the quota holds it')

    def test_the_bound_re_enters_the_owner(self):
        """The owner's starvation bound: at yield_window_max appends the owner re-enters although appends wait;
        the next waiting append gets a later window (its own PER_BULK count)."""
        rig = window_rig()
        rig.c.yield_window_max = 2
        append, (other, third) = served_first(rig, (OTHER_UID, THIRD_UID))
        frontier = rig.bulk.num_computed_tokens
        to_window_append(rig)
        self.assertIs(rig.d._yield['append'], other)
        ay.run_append(rig)
        ay.to_full(rig)
        rig.stock_handback(other)
        ay.to_resume(rig)
        self.assertIn(third, rig.s.skipped_waiting.items, 'the third append waits: the bound held')
        txn = rig.admit()
        self.assertEqual((txn.request_uid, txn.chunk_start), (BULK_UID, frontier))
        rig.finish_chunk()
        self.assertEqual(rig.d.append_yield_phase(), 'closing', 'a new window for the third at the next junction')
        self.assertIs(rig.d._yield['append'], third)
        w = counts(rig)
        self.assertEqual((w['per_window'], w['ends']), ({'2': 1}, {'bound': 1}))
        self.assertEqual(rig.d.append_yield_counters['selected'], 2)

    def test_the_default_bound_is_four(self):
        rig = window_rig()
        append, held = served_first(rig, (OTHER_UID, THIRD_UID, FOURTH_UID, 'yw-fifth'))
        order = []
        for _ in range(3):
            to_window_append(rig)
            current = rig.d._yield['append']
            order.append(current.request_id)
            ay.run_append(rig)
            ay.to_full(rig)
            rig.stock_handback(current)
        ay.to_resume(rig)
        self.assertEqual(order, [OTHER_UID, THIRD_UID, FOURTH_UID], 'FCFS inside the window')
        self.assertEqual(counts(rig)['per_window'], {})
        ay.finish_bulk(rig)
        w = counts(rig)
        self.assertEqual((w['per_window'], w['ends'], w['reentries_avoided'], w['gap_ms']['n']),
                         ({'4': 1}, {'bound': 1}, 3, 3))

    def test_kv_block_gate_is_per_candidate(self):
        """The next append must fit now (the candidate's own free-block check); refused, the owner re-enters."""
        rig = window_rig()
        append, (other,) = served_first(rig)
        rig.free_blocks = 3                                   # OTHER needs 6 blocks
        ay.to_resume(rig)
        w = rig.d._yield['window']
        self.assertEqual(w['end'], 'none-waiting')
        self.assertEqual(rig.c.yield_refusals.get('blocks'), 1)

    def test_decode_cover_gate_is_per_candidate(self):
        rig = window_rig()
        append, (other,) = served_first(rig)
        rig.p.decode_counts = frozenset({1})                  # APPEND is a decoder now: cohort 2 not proven
        ay.to_resume(rig)
        self.assertEqual(rig.d._yield['window']['end'], 'none-waiting')
        self.assertGreaterEqual(rig.c.yield_refusals.get('decode-cover', 0), 1)


class TailAppend(unittest.TestCase):
    """A geometry-tail append hands back r > 1 rows; the next append may run split only after that FULL handback."""

    def setUp(self):
        self.log = Lines(self)

    def tail_rig(self, o3):
        rig = lo3.yield_rig() if o3 else window_rig()
        rig.c.yield_window = True
        rig.s.max_num_running_reqs = 8
        rig.c.append_yield = sh.AppendYieldConfig(8, 4, 8, 4, enabled=True)
        geometry = {APPEND_UID: 2}
        rig.c.geometry_tail = lambda scheduler, request: geometry.get(request.request_id, 0)
        return rig

    def run_tail(self, o3):
        rig = self.tail_rig(o3)
        d = rig.d
        append = ay.run_to_yield(rig)
        other = rig.add_held(OTHER_UID, 6)
        ay.to_append(rig)
        transactions = ay.run_append(rig)
        self.assertEqual(transactions[-1].chunk_end, 9, 'native ends at the block boundary: 2 rows for stock')
        self.assertEqual(d.append_yield_phase(), 'handback')
        if o3:
            self.assertEqual(rig.p.o3_counts['yield_split'], 0, 'O3 never keeps a tail split')
        ay.to_full(rig)
        ay.policy_ticks(rig, 3)                               # the FULL window: the tail not yet handed back
        self.assertEqual(d.append_yield_phase(), 'handback')
        self.assertIsNone(rig.admit(), 'no other append before the FULL handback')
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, append), 'the tail rows pass on FULL')
        rig.stock_handback(append)                            # the r rows and the first token, on FULL
        to_window_append(rig)
        self.assertIs(d._yield['append'], other)
        self.assertTrue(d._split_ready)
        ay.run_append(rig)
        w = counts(rig)
        self.assertEqual((w['tail_full_handbacks'], w['reentries_avoided']), (1, 1))
        self.assertTrue(rig.journal('append_yield_window_next')[-1]['tail_handback'])
        return rig

    def test_tail_append_full_handback_first(self):
        self.run_tail(o3=False)

    def test_tail_append_full_handback_first_under_o3(self):
        self.run_tail(o3=True)


class AbortInWindow(unittest.TestCase):
    def setUp(self):
        self.log = Lines(self)

    def test_abort_of_the_second_append_mid_chunk_ends_the_window(self):
        """Client exit inside the window: normal native cancel, FULL; the window ends (not served) although THIRD
        waits, and the owner re-enters and reopens from the window's frontier."""
        rig = window_rig()
        d = rig.d
        append, (other, third) = served_first(rig, (OTHER_UID, THIRD_UID))
        frontier = rig.bulk.num_computed_tokens
        to_window_append(rig)
        self.assertIsNotNone(rig.admit())
        rig.drive(lambda: d.tx is not None)
        self.assertIsNotNone(ay.abort(rig, other))
        rig.stock_abort(other)
        self.assertEqual(d.append_yield_phase(), 'handback')
        ay.settle_stop(rig)
        ay.to_resume(rig)
        self.assertIn(third, rig.s.skipped_waiting.items)
        txn = rig.admit()
        self.assertEqual((txn.request_uid, txn.chunk_start), (BULK_UID, frontier))
        rig.finish_chunk()
        self.assertEqual(rig.journal('append_yield_append_aborted')[-1]['seq'], 1)
        self.assertEqual(rig.d._yield_history[-1]['window'], dict(appends=2, end='not-served'))
        self.assertEqual(counts(rig)['ends'], {'not-served': 1})

    def test_abort_on_full_before_the_handback_ends_the_window(self):
        rig = window_rig()
        append = ay.run_to_yield(rig)
        third = rig.add_held(THIRD_UID, 6)
        ay.to_append(rig)
        ay.run_append(rig)
        ay.to_full(rig)
        self.assertIsNone(ay.abort(rig, append))
        rig.stock_abort(append)
        ay.to_resume(rig)
        self.assertEqual(rig.d._yield['window']['end'], 'not-served')
        self.assertIn(third, rig.s.skipped_waiting.items)

    def test_bulk_abort_inside_the_window(self):
        """The owner leaves while the window's second append runs: the append completes natively, no re-entry."""
        rig = window_rig()
        append, (other,) = served_first(rig)
        to_window_append(rig)
        self.assertIsNone(ay.abort(rig, rig.bulk))
        rig.stock_abort(rig.bulk)
        ay.run_append(rig)
        ay.to_full(rig)
        rig.stock_handback(other)
        ay.policy_idle(rig)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'bulk-left')
        self.assertEqual(rig.d._yield_history[-1]['window'], dict(appends=2, end='yield-bulk-left'))
        self.assertFalse(rig.d._split_ready)

    def test_the_staged_append_leaves_before_the_arm(self):
        rig = window_rig()
        append, (other,) = served_first(rig)
        frontier = rig.bulk.num_computed_tokens
        rig.tick(lambda: rig.p.phase == 'split')              # staged; the re-entry's enter_split in flight
        self.assertIs(rig.d._yield['window']['next'], other)
        self.assertIsNone(ay.abort(rig, other))
        rig.stock_abort(other)
        ay.to_resume(rig)
        self.assertEqual(rig.d._yield['window']['end'], 'next-left')
        self.assertIn('window_next_left', kinds(rig))
        txn = rig.admit()
        self.assertEqual((txn.request_uid, txn.chunk_start), (BULK_UID, frontier))


class O3Window(unittest.TestCase):
    """O3 installed and selected: the first append's one-row handback stays split; the census arms the window's next
    append on that split route (no FULL transition), as it would arm the owner's resume."""

    def setUp(self):
        self.log = ts.Log()
        patcher = mock.patch.object(ed, '_progress_logger', lambda: self.log)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_next_append_on_the_kept_split_route(self):
        rig = lo3.yield_rig()
        rig.c.yield_window = True
        rig.s.max_num_running_reqs = 8
        p, d = rig.p, rig.d
        append = ay.run_to_yield(rig)
        other = rig.add_held(OTHER_UID, 6)
        frontier = rig.bulk.num_computed_tokens
        ay.to_append(rig)
        ay.run_append(rig)
        generation, moves = d.route_generation, len(lo3.transitions(rig))
        self.assertEqual(p.o3_counts['yield_split'], 1)
        ay.policy_ticks(rig, 1)
        self.assertEqual((p.phase, p.o3_yield), ('tail', True))
        lo3.split_handback_step(rig, append)
        lo3.worker_lag(rig, append)
        ay.policy_ticks(rig, 1)
        lo3.first_decode_step(rig, append)
        ay.policy_ticks(rig, 1)
        self.assertEqual(p.phase, 'o3_drain')
        self.assertIs(d._yield['window']['next'], other)
        ts.landed(rig, append)
        rig.tick(lambda: d.append_yield_phase() == 'append')
        self.assertIs(d._yield['append'], other)
        self.assertTrue(d._split_ready)
        self.assertEqual(d.route_generation, generation, 'the route never left split')
        self.assertEqual(len(lo3.transitions(rig)), moves)
        self.assertEqual(p.o3_counts['yield_kept'], 1)
        ay.run_append(rig)
        self.assertEqual(d.append_yield_counters['admitted'], 2)
        self.assertEqual(rig.c.yield_window_counts['reentries_avoided'], 1)
        self.assertEqual(rig.c.yield_window_counts['tail_full_handbacks'], 0)
        self.assertEqual(d._yield['frontier'], frontier)


def stock_handback_unsampled(rig, request):
    """Leaf (night-pack fix 5): the box's FULL handback with asynchronous sampling. The pinned stock step scheduled the
    request's last prompt token on FULL (prompt fully computed, a live decode row on every worker) but its first token
    is not yet sampled when the FULL window ends; rig.stock_handback samples at once, which is why the CPU rig never
    saw the as-built 'not-served' end on the FULL path (V1: 408 of 659 handbacks served = the O3 count)."""
    rig.stock_handback(request)
    request.num_output_tokens = 0


class AsBuiltReceipts(unittest.TestCase):
    """night-pack fix 5: receipts only; Y2's behaviour is pinned as built (not changed)."""

    def setUp(self):
        self.log = Lines(self)

    def test_unsampled_full_handback_ends_the_window_although_an_append_waits(self):
        rig = window_rig()
        d = rig.d
        append = ay.run_to_yield(rig)
        other = rig.add_held(OTHER_UID, 6)
        ay.to_append(rig)
        ay.run_append(rig)
        ay.to_full(rig)
        stock_handback_unsampled(rig, append)
        frontier = rig.bulk.num_computed_tokens
        ay.to_resume(rig)                                   # as built: the owner re-enters, OTHER keeps waiting
        w = d._yield['window']
        self.assertEqual((w['end'], w['not_served'], w['offer_path']), ('not-served', 'unsampled', 'full'))
        self.assertIsNone(w['next'])
        self.assertEqual(d.append_yield_counters['handbacks_served'], 0)
        self.assertIn(other, rig.s.skipped_waiting.items)
        append.num_output_tokens = 1                        # the delayed sample lands: nothing moves
        txn = rig.admit()
        self.assertEqual((txn.request_uid, txn.chunk_start), (BULK_UID, frontier))
        rig.finish_chunk()
        c = counts(rig)
        self.assertEqual((c['ends'], c['not_served'], c['per_window'], c['reentries_avoided']),
                         ({'not-served': 1}, {'unsampled': 1}, {'1': 1}, 0))
        closed = [r for r in self.log.receipts() if r['event'] == 'closed']
        self.assertEqual([(r['end'], r['not_served'], r['appends']) for r in closed], [('not-served', 'unsampled', 1)])
        block = rig.c.native_admission_record()['queue_policy']['yield_window']
        self.assertEqual(block['not_served'], {'unsampled': 1})

    def test_the_same_flow_sampled_continues_the_window(self):
        """Control: sampled at once (the rig's default leaf) the same flow takes OTHER in the window."""
        rig = window_rig()
        append, (other,) = served_first(rig)
        to_window_append(rig)
        self.assertIs(rig.d._yield['append'], other)
        self.assertEqual(counts(rig)['not_served'], {})

    def test_abort_is_split_as_abort(self):
        rig = window_rig()
        append = ay.run_to_yield(rig)
        rig.add_held(THIRD_UID, 6)
        ay.to_append(rig)
        ay.run_append(rig)
        ay.to_full(rig)
        self.assertIsNone(ay.abort(rig, append))
        rig.stock_abort(append)
        ay.to_resume(rig)
        self.assertEqual((rig.d._yield['window']['end'], rig.d._yield['window']['not_served']), ('not-served', 'abort'))
        self.assertEqual(counts(rig)['not_served'], {'abort': 1})

    def test_append_receipt_path_full_and_gap_by_path(self):
        rig = window_rig()
        append, (other,) = served_first(rig)
        to_window_append(rig)
        ay.run_append(rig)
        appends = [r for r in self.log.receipts() if r['event'] == 'append']
        self.assertEqual([(r['index'], r['path']) for r in appends], [(1, None), (2, 'full')])
        g = counts(rig)['gap_by_path']
        self.assertEqual(list(g), ['full'])
        self.assertEqual(g['full']['n'], 1)

    def test_staged_candidate_that_left_is_counted(self):
        rig = window_rig()
        append, (other,) = served_first(rig)
        rig.tick(lambda: rig.p.phase == 'split')
        self.assertIs(rig.d._yield['window']['next'], other)
        self.assertIsNone(ay.abort(rig, other))
        rig.stock_abort(other)
        ay.to_resume(rig)
        self.assertEqual(rig.d._yield['window']['end'], 'next-left')
        self.assertEqual(counts(rig)['next_left'], 1)

    def test_o3_path_on_the_kept_split_route(self):
        log = ts.Log()
        with mock.patch.object(ed, '_progress_logger', lambda: log):
            rig = lo3.yield_rig()
            rig.c.yield_window = True
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
            self.assertEqual(d._yield['window']['offer_path'], 'o3')
            ts.landed(rig, append)
            rig.tick(lambda: d.append_yield_phase() == 'append')
            ay.run_append(rig)
        self.assertEqual(d._yield['window']['path'], 'o3')
        self.assertEqual(list(rig.c.yield_window_counts['gap_by_path']), ['o3'])


class OffIsFrozen(unittest.TestCase):
    """Flag at 0: no window record, no window call (offer/begin raise if reached), the same journal and counters."""

    def setUp(self):
        Lines(self)

    def run_cycle(self, rig):
        append, (other,) = served_first(rig)
        ay.to_resume(rig)
        rig.admit()
        rig.finish_chunk()
        ay.to_append(rig)                                     # today: the other append's own yield at frontier 8
        ay.run_append(rig)
        ay.to_full(rig)
        rig.stock_handback(other)
        ay.to_resume(rig)
        ay.finish_bulk(rig)
        return dict(kinds=kinds(rig), counters=dict(rig.d.append_yield_counters), lookups=list(rig.lookups),
                    history=[{k: v for k, v in h.items() if k != 'times'} for h in rig.d._yield_history])

    def test_flag_off_never_calls_the_window(self):
        boom = mock.Mock(side_effect=AssertionError('window call with the flag off'))
        with mock.patch.object(ed.NativeSegmentDriver, 'append_yield_window_offer', boom), \
                mock.patch.object(ed.NativeSegmentDriver, 'append_yield_window_begin', boom):
            rig = window_rig(on=False)
            off = self.run_cycle(rig)
            self.assertNotIn('window', rig.d._yield_history[-1])
        one = ['selected', 'drained', 'row_closed', 'admitted', 'handback', 'full', 'reenter', 'resume', 'bulk_reopened',
               'end']
        self.assertEqual(off['kinds'], one + one, "today's two yields, one append each")
        self.assertEqual([h['reason'] for h in off['history']], ['reopened', 'reopened'])
        self.assertEqual((off['counters']['selected'], off['counters']['admitted']), (2, 2))
        self.assertNotIn('yield_window', rig.c.native_admission_record().get('queue_policy') or {})

    def test_flag_off_o3_path_never_calls_the_window(self):
        boom = mock.Mock(side_effect=AssertionError('window call with the flag off'))
        with mock.patch.object(ed.NativeSegmentDriver, 'append_yield_window_offer', boom), \
                mock.patch.object(ed.NativeSegmentDriver, 'append_yield_window_begin', boom), \
                mock.patch.object(ed, '_progress_logger', lambda: ts.Log()):
            lo3.YieldSplit('test_the_append_handback_stays_split_and_the_bulk_resumes_on_it') \
                .test_the_append_handback_stays_split_and_the_bulk_resumes_on_it()


class WindowStaged(ay.StagedDriver, Window):
    pass


class AbortInWindowStaged(ay.StagedDriver, AbortInWindow):
    pass


class TailAppendStaged(ay.StagedDriver, TailAppend):
    pass


class Mutants(unittest.TestCase):
    """Each rule's mutant must fail its test (the method's own source, mutated, bound in the driver module)."""

    def setUp(self):
        Lines(self)

    def run_mutant(self, owner, name, old, new, test, module=ed):
        source = inspect.getsource(getattr(owner, name))       # the method as in its class (4-space body indent)
        self.assertEqual(source.count(old), 1, old)
        namespace = dict(module.__dict__)
        exec(compile('class _Mutant:\n' + source.replace(old, new, 1), module.__file__, 'exec'), namespace)
        with mock.patch.object(owner, name, namespace['_Mutant'].__dict__[name]):
            with self.assertRaises(AssertionError):
                test()

    def case(self, cls, method):
        instance = cls(method)
        instance.setUp()
        self.addCleanup(instance.doCleanups)
        return getattr(instance, method)

    def test_no_bound_caught(self):
        self.run_mutant(ed.NativeSegmentDriver, 'append_yield_window_offer',
                        "            if w['appends'] >= c.yield_window_max:\n",
                        "            if False:\n", self.case(Window, 'test_the_bound_re_enters_the_owner'))

    def test_unserved_handback_continues_caught(self):
        self.run_mutant(ed.NativeSegmentDriver, 'append_yield_window_offer',
                        "            if why is not None:\n",
                        "            if False:\n",
                        self.case(AbortInWindow, 'test_abort_of_the_second_append_mid_chunk_ends_the_window'))

    def test_unsampled_continues_caught(self):
        """night-pack fix 5: the as-built pin has power (an offer that continued after an unsampled FULL handback)."""
        self.run_mutant(ed.NativeSegmentDriver, 'append_yield_window_offer',
                        "            if why is not None:\n",
                        "            if False:\n",
                        self.case(AsBuiltReceipts, 'test_unsampled_full_handback_ends_the_window_although_an_append_waits'))

    def test_in_window_per_bulk_count_caught(self):
        self.run_mutant(ed.NativeSegmentDriver, 'append_yield_window_begin',
                        "        w['appends'] += 1\n",
                        "        w['appends'] += 1\n        self._yield_counts[y['key']] = self._yield_counts.get(y['key'], 0) + 1\n",
                        self.case(Window, 'test_two_appends_one_window_then_the_owner'))

    def test_tail_uncounted_caught(self):
        self.run_mutant(ed.NativeSegmentDriver, 'append_yield_window_offer',
                        "            tail = self._handback_rows(y['append']) > 1\n",
                        "            tail = False\n", self.case(TailAppend, 'test_tail_append_full_handback_first'))

    def test_offer_before_the_served_note_caught(self):
        """The FULL path must note the served handback before the offer (else every window ends 'not-served')."""
        self.run_mutant(auto.AutoArmPolicy, '_yield_native',
                        "        d.append_yield_note_handback()\n        if self._yield_window():\n"
                        "            d.append_yield_window_offer()                            # yield-rules Y2: off, never called\n",
                        "        if self._yield_window():\n"
                        "            d.append_yield_window_offer()                            # yield-rules Y2: off, never called\n"
                        "        d.append_yield_note_handback()\n",
                        self.case(Window, 'test_two_appends_one_window_then_the_owner'), module=auto)


if __name__ == '__main__':
    unittest.main()
