"""compose-5's table, box binding and rule. CPU only.

compose-4's rule tests run again over this table, through the wrapper, with ny as the control: what held for ctl has to hold for ny. On top, the frozen pair's price and the r1 read over known numbers.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import compose4_arms as T4
import compose4_rule
import compose5_arms as T
import compose5_rule as rule
import e3_compose4_box as box4
import e3_compose5_box as box
import test_compose4_box as tb

tam, tap, ta1 = tb.tam, tb.tap, tb.ta1
SM, DEC1, CMP, NY = rule.SM, 'dec1.g2-live-0', rule.ON, rule.OFF      # the rule's labels: compose-5's, for its own suites; dnr's table has no cmp
N22, NY1, N221 = 'n22.g2-live-0', 'ny.g2-live-1', 'n22.g2-live-1'
PAIRS = rule.PRICE_PAIRS


def frozen(*, bulk=20.0, decode=56.0, returners=((500, 1.5), (1500, 2.5)), selected=2):
    return dict(client_basis=dict(bulk_complete_s=bulk, prefill_tokens_per_s=65445 / bulk, decode_window=dict(aggregate_tokens_per_s=8 * decode, stream_rates_tokens_per_s=[decode] * 8, gap_ms_p50=20.0, gap_ms_p99=70.0, gap_ms_max=500.0)),
                statuses=[dict(label='after', append_yield=dict(counters=dict(selected=selected)))], returners=[dict(suffix_tokens=s, ttft_s=t) for s, t in returners])


class Fixture5(tb.Fixture):
    """compose-4's record dir with the frozen arms this table adds, whole and quiet."""
    def __init__(self, tmp):
        super().__init__(tmp)
        for on, off in PAIRS:
            self.write_record(on, wall_start_s=tb.ORIGIN - 200, wall_end_s=tb.ORIGIN - 150, **frozen())
            self.write_record(off, wall_start_s=tb.ORIGIN - 140, wall_end_s=tb.ORIGIN - 90, **frozen(selected=0))


class UnderThisTable(unittest.TestCase):
    def setUp(self):
        for name, value in (('CTL', NY), ('rule', rule), ('Fixture', Fixture5)):
            patcher = mock.patch.object(tb, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        super().setUp()


class Rule(UnderThisTable, tb.Rule):
    """Every rule test of compose-4's, here over the wrapper, whose module hands them compose-4's helpers."""


class StoppedArm(UnderThisTable, tb.StoppedArm):
    pass


class JournalAgainstRecord(UnderThisTable, tb.JournalAgainstRecord):
    def test_a_missing_journal_is_inc_on_either_arm(self):   # compose-4's names the old control's file
        healthy_turns = self.fx.turns[CMP]
        self.assertTrue(healthy_turns)
        self.fx.done()
        path = self.fx.root / ('arm-%s.json.turns.jsonl' % NY)
        path.rename(path.with_suffix('.moved'))
        word, _lines, facts = rule.report(str(self.fx.root))
        self.assertEqual(word, 'INC')
        self.assertEqual(facts['arms'][NY]['journal_problem'], 'journal-missing-or-no-header')
        self.assertIn('journal: ' + NY, ' '.join(facts['reasons']))


class Table(unittest.TestCase):
    def test_the_order(self):
        self.assertEqual(T.labels()[:6], [SM, DEC1, NY, N22, N221, NY1])
        self.assertEqual(len(T.labels()), 22)
        self.assertEqual(T.labels()[6:8], ['ny.g2-live-2', 'n22.g2-live-2'])
        self.assertEqual(T.labels()[-2:], ['n22.g2-live-9', 'ny.g2-live-9'])

    def test_each_pair_differs_in_the_plan_alone_and_shares_a_seed(self):
        rows = T.by_label()
        for on, off, seed in tuple(('n22.g2-live-%d' % i, 'ny.g2-live-%d' % i, i + 1) for i in range(10)):
            self.assertEqual(sorted(k for k in rows[on] if rows[on][k] != rows[off][k]), ['c1', 'chunk_bound', 'label', 'selection'], (on, off))
            self.assertEqual((rows[on]['chunk_bound'], rows[off]['chunk_bound']), ([89056, 1024], [52624, 1024]))
            self.assertEqual((T.replay_config(rows[on])['SEED'], T.replay_config(rows[off])['SEED']), (seed, seed))

    def test_a_replays_config_is_compose_4s_but_for_the_seed(self):
        row = T.by_label()[N221]
        self.assertEqual({k: v for k, v in T.replay_config(row).items() if T4.REPLAY[k] != v}, dict(SEED=2))
        self.assertEqual(T4.REPLAY['SEED'], 1, "the table's own stays as it is")

    def test_the_cap_is_off_on_every_arm(self):
        self.assertEqual({r['reserved_cap'] for r in T.arms()}, {False})

    def test_the_private_ple_buffer_only_off_on_the_replays_and_the_witness_on_none(self):
        for row in T.arms():
            self.assertEqual((row['ple_private'], row['ple_witness']), (row['kind'] != 'replay', False), row['label'])

    def test_compose_4s_ctl_is_the_base(self):
        c4 = T4.by_label()['ctl.g2-live-0']
        row = T.by_label()[NY]
        self.assertEqual(sorted(k for k in row if k not in c4 or row[k] != c4[k]), ['c1', 'chunk_bound', 'label', 'reserved_cap', 'selection'])
        self.assertEqual((row['chunk_bound'], row['reserved_cap'], row['yield_on'], row['ple_private']), ([52624, 1024], False, False, False))

    def test_the_replay_and_the_ceiling_are_compose_4s(self):
        self.assertIs(T.REPLAY, T4.REPLAY)
        self.assertEqual(T.MAX_PROMPT_TOKENS, 135000)
        self.assertEqual((T.RULE_PREFILL_TOK_S, T.RULE_DECODE_VS_FIXED, T.RULE_MIN_NATIVE_BULKS), (T4.RULE_PREFILL_TOK_S, T4.RULE_DECODE_VS_FIXED, T4.RULE_MIN_NATIVE_BULKS))

    def test_a_replays_request_count_is_in_its_record(self):
        for label in (NY, N22, N221, NY1):
            self.assertIsNone(T.by_label()[label]['generations'])

    def test_a_matched_off_only_for_the_on_arms(self):
        rows = T.by_label()
        self.assertEqual(T.matched_off(rows[N221]), NY1)
        self.assertEqual(T.matched_off(rows[N22]), NY)
        self.assertIsNone(T.matched_off(rows[SM]))


class CapReadBack(unittest.TestCase):
    def test_the_real_records(self):
        def rec(state):
            return dict(reserved_cap=dict(select={str(k): dict(after=dict(reserved_cap=state), before={}, order={}, counters={}) for k in range(4)}))
        self.assertIs(box.cap_selected(rec(dict(cap_bytes=0, flushes=0, fraction=0.0, selected=False))), False, "dn's")
        self.assertIs(box.cap_selected(rec(dict(cap_bytes=23369547776, flushes=1, fraction=0.92, overdraft_refused=0, overdrafts=0, selected=True))), True, "c4b's")
        self.assertIsNone(box.cap_selected(dict()))


class ReplaySeed(unittest.TestCase):
    """The seed reaches the run and the journal header, by the real replay_arm of compose-4's box up to the run call."""

    def go(self, label):
        row = T.by_label()[label]
        seen = {}

        class Stop(Exception):
            pass

        def fake(a, row_, client, poll_client, *, run=None):
            seen['header'] = box4.T.REPLAY
            try:
                run('u', 'm', 'k', 'r', row_['label'], sink=None)
            except Stop:
                pass
            return dict(label=row_['label'])

        def real_run(*args, cfg=None, **kwargs):
            seen['cfg'] = cfg
            raise Stop()
        with mock.patch.object(box, '_replay_arm', fake), mock.patch.object(box._box.replay, 'run', real_run):
            box.replay_arm(None, row, None, None)
        return seen

    def test_each_index_gets_its_seed_and_the_table_is_put_back(self):
        saved = T.REPLAY
        for label, seed in ((NY, 1), (N22, 1), (N221, 2), (NY1, 2)):
            seen = self.go(label)
            self.assertEqual((seen['cfg']['SEED'], seen['header']['SEED']), (seed, seed), label)
        self.assertIs(T.REPLAY, saved)
        self.assertIs(box4.T, T4)


class Smoke(unittest.TestCase):
    """What this table adds to compose-4's smoke; b13h's numbers in the record are dp8d's (q18a-dp8-c32, arm-b13h.g2-live-0.json)."""

    def full(self):
        row = T.by_label()[SM]
        rec = ta1.record_of(row)
        rec.update(ple_private=dict(counters=dict(scopes={'0': 9, '1': 9})), score_warm=dict(counters=dict(allocs={str(k): 1 for k in range(4)})), reserved_cap=dict(requested=False, select={str(k): dict(after=dict(reserved_cap=dict(cap_bytes=0, flushes=0, fraction=0.0, selected=False))) for k in range(4)}, counters=dict(flushes={str(k): 0 for k in range(4)})),
                   cow_host_filter=dict(requested=True, counters=dict(kept={str(k): 0 for k in range(4)}, skipped={str(k): 37 for k in range(4)}, unfiltered={str(k): 0 for k in range(4)})))
        rec.setdefault('inwait', {}).update(exact_acknowledged=True, bound_requested=[52624, 1024], bound_acknowledged=[52624, 1024])
        rec.setdefault('progress_delta', {})['chunks'] = 40
        rec.setdefault('fresh', {})['prompt_tokens_observed'] = 65445
        return rec

    def assert_stops(self, change, reason):
        rec = self.full()
        change(rec)
        ok, reasons = box.smoke_check(rec)
        self.assertFalse(ok)
        self.assertIn(reason, reasons)

    def test_dp8ds_numbers_pass(self):
        self.assertEqual(box.smoke_check(self.full()), (True, []))

    def test_the_cap_selected_stops_the_table(self):
        self.assert_stops(lambda r: r['reserved_cap'].update(requested=True), 'smoke-reserved-cap-not-off')
        self.assert_stops(lambda r: r.pop('reserved_cap'), 'smoke-reserved-cap-not-off')
        # The numbers are c4b's sm on r1 and dn's sm, the real shape of the rank's answer.
        self.assert_stops(lambda r: r['reserved_cap']['select'].update({'1': dict(after=dict(reserved_cap=dict(cap_bytes=23369547776, flushes=1, fraction=0.9238, selected=True)))}), 'smoke-reserved-cap-selected')
        self.assert_stops(lambda r: r['reserved_cap']['select'].pop('3'), 'smoke-reserved-cap-selected')
        self.assert_stops(lambda r: r['reserved_cap'].update(select=None), 'smoke-reserved-cap-selected')

    def test_the_bound_not_acknowledged_stops_the_table(self):
        self.assert_stops(lambda r: r['inwait'].update(bound_acknowledged=None), 'smoke-chunk-bound-not-acknowledged')

    def test_a_launch_that_fell_back_to_1024_stops_the_table(self):
        # compose-4's smoke at 1024: 71 chunks, its 66 and the 5 of its returners and append. That's the control: under b13, with the same 5, 45.
        self.assert_stops(lambda r: r['progress_delta'].update(chunks=71), 'smoke-chunk-rows-not-requested')
        self.assert_stops(lambda r: r['progress_delta'].update(chunks=66), 'smoke-chunk-rows-not-requested')
        rec = self.full()
        rec['progress_delta']['chunks'] = 45
        self.assertEqual(box.smoke_check(rec), (True, []))

    def test_no_count_to_read_stops_the_table(self):
        self.assert_stops(lambda r: r['progress_delta'].pop('chunks'), 'smoke-chunk-rows-not-requested')
        self.assert_stops(lambda r: r['fresh'].pop('prompt_tokens_observed'), 'smoke-chunk-rows-not-requested')

    def test_the_host_filter_not_engaged_on_a_rank_stops_the_table(self):
        self.assert_stops(lambda r: r['cow_host_filter']['counters']['skipped'].update({'2': 0}), 'smoke-host-filter-not-engaged')
        self.assert_stops(lambda r: r['cow_host_filter'].update(counters={}), 'smoke-host-filter-not-engaged')
        self.assert_stops(lambda r: r['cow_host_filter']['counters']['unfiltered'].update({'3': 1}), 'smoke-host-filter-unfiltered')


REAL_TORCH = (" (Worker_PP0_TP0_EP0 pid=96) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 194.00 MiB. GPU 0 has a total capacity of 23.56 GiB of which 164.94 MiB is free. Including non-PyTorch memory\n",
              " (Worker_PP0_TP1_EP1 pid=102) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 194.00 MiB. GPU 1 has a total capacity of 23.56 GiB of which 60.94 MiB is free. Including non-PyTorch memory\n")     # dp8d's cf, band 2


class CrashRead(UnderThisTable, unittest.TestCase):
    """A read on an engine death, never the verdict: compose-4's clause already says FAIL."""

    def fixture(self, tmp, *, death_at=None, turns=True):
        fx = Fixture5(tmp)
        tb.healthy(fx)
        tb.healthy(fx, NY)
        if not turns:
            fx.turns[CMP] = []
        if death_at is not None:
            fx.lines.append(tb.memory_line(death_at - 1, 'open', 1, 900, 97152, 99200, free=60))
            fx.lines.append(tb.memory_line(death_at - 1, 'open', 0, 900, 97152, 99200, free=164))
            (fx.root / 'engine-stream.err').write_text(''.join(tb.stamp(death_at) + line for line in REAL_TORCH))
        return fx

    def test_a_clean_run_prints_nothing_of_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            _word, lines, facts = self.fixture(tmp).done()
        self.assertFalse([x for x in lines if x.startswith('CRASH')])
        self.assertIsNone(facts['crash'])

    def test_a_death_names_the_arm_each_ranks_chunk_and_torchs_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            word, lines, facts = self.fixture(tmp, death_at=tb.ORIGIN + 100).done()
        self.assertEqual(word, 'FAIL', 'the verdict is the old clause')
        crash = facts['crash']
        self.assertEqual(crash['arm'], CMP)
        self.assertEqual({k: crash['ranks'][k]['chunk'] for k in ('0', '1')}, {'0': '97152-99200', '1': '97152-99200'})
        self.assertEqual(crash['tried']['0']['tried_mib'], 194.0)
        self.assertEqual(crash['tried']['1']['free_mib'], 60.94)
        self.assertEqual(crash['tried']['0']['position_if_score_at_2048'], 99328, "dp8d's chunk ended at 99200: the width rounds up to 64")
        self.assertEqual(crash['tried']['0']['position_if_score_at_1024'], 198656)
        self.assertGreater(crash['turns']['finished'], 0)
        self.assertTrue([x for x in lines if x.startswith('CRASH GPU 1')])

    def test_the_verdict_is_the_same_with_the_read_or_without(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(rule, 'crash_read', return_value=None):
            without = self.fixture(tmp, death_at=tb.ORIGIN + 100).done()[0]
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.fixture(tmp, death_at=tb.ORIGIN + 100).done()[0], without)

    def test_a_death_between_arms_says_after_which(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self.fixture(tmp)
            record = json.loads((fx.root / ('arm-%s.json' % PAIRS[0][0])).read_text())
            self.assertEqual(rule.arm_at(str(fx.root), (PAIRS[0][0],), record['wall_end_s'] + 5), ('after ' + PAIRS[0][0], None))

    def test_a_line_without_a_size_or_a_stamp_is_kept_going(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self.fixture(tmp)
            (fx.root / 'engine-stream.err').write_text('garbage OutOfMemoryError\n' + tb.stamp(tb.ORIGIN) + ' FLASH_S5_WORKER_FAILURE rank=1 cause=OutOfMemoryError\n')
            crash = rule.crash_read(str(fx.root))
        self.assertEqual(crash['tried'], {})
        self.assertIsNotNone(crash['first_death_s'])


class JsonCarriesTheReasons(UnderThisTable, unittest.TestCase):
    """Codex, REVIEW-compose5 item 1: a word this rule changes has its reason in the JSON too, not only in the text."""

    def over_the_bound(self, tmp):
        fx = Fixture5(tmp)
        tb.healthy(fx)
        tb.healthy(fx, NY)
        fx.lines.append(tb.memory_line(tb.ORIGIN + 50, 'open', 0, 9000, 52624, 54672))
        return fx.done()

    def test_a_chunk_past_the_bound_is_inc_with_its_reason_in_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            word, lines, facts = self.over_the_bound(tmp)
        self.assertEqual((word, facts['rule']), ('INC', 'INC'))
        self.assertEqual(facts['reasons'], ['chunk-rows: 1 chunks past the bound over 1024 rows'])
        self.assertIn('REASON ' + facts['reasons'][0], lines)

    def test_a_stock_rescue_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture5(tmp)
            tb.healthy(fx)
            tb.healthy(fx, NY)
            with mock.patch.object(rule, 'stock_rescues', return_value=2):
                word, lines, facts = fx.done()
        self.assertEqual(word, 'INC')
        self.assertEqual(facts['reasons'], ['stock-rescue: cmp 2'])
        self.assertIn('REASON stock-rescue: cmp 2', lines)

    def test_the_base_reasons_stay_and_a_clean_run_has_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture5(tmp)
            tb.healthy(fx)
            tb.healthy(fx, NY)
            word, _lines, facts = fx.done()
        self.assertEqual((word, facts['reasons']), ('PASS', []))
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture5(tmp)
            tb.healthy(fx)
            tb.healthy(fx, NY)
            fx.lines.append(tb.memory_line(tb.ORIGIN + 50, 'open', 0, 9000, 52624, 54672))
            fx.lines.append(tb.stamp(tb.ORIGIN) + ' ERROR torch.OutOfMemoryError: CUDA out of memory\n')
            word, _lines, facts = fx.done()
        self.assertEqual(word, 'FAIL', 'INC never lowers a FAIL')
        self.assertTrue(any('OutOfMemoryError' in x for x in facts['reasons']))
        self.assertIn('chunk-rows: 1 chunks past the bound over 1024 rows', facts['reasons'])


class ChunkRowsOfADeadArm(UnderThisTable, unittest.TestCase):
    """Codex, REVIEW-compose5 item 2: a crashed cmp has no final record; its count is partial or unavailable, never zero."""

    def crashed(self, tmp, *, death=True):
        fx = Fixture5(tmp)
        tb.healthy(fx, starts=(320, 500, 700))
        fx.lines.append(tb.memory_line(tb.ORIGIN + 900, 'open', 0, 9000, 97152, 98176))
        fx.lines.append(tb.memory_line(tb.ORIGIN + 1000, 'open', 0, 9001, 97152, 98176))      # past the death line: not this arm's
        (fx.root / 'engine-stream.log').write_text(''.join(fx.lines))
        fx.write_turns(CMP)
        if death:
            (fx.root / 'engine-stream.err').write_text(tb.stamp(tb.ORIGIN + 950) + REAL_TORCH[0])
        return rule.report(str(fx.root))

    def test_the_stopped_arm_counts_its_real_rows_marked_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            word, lines, facts = self.crashed(tmp)
        self.assertEqual(word, 'FAIL')
        got = facts['chunk_rows'][CMP]
        self.assertIs(got['partial'], True)
        self.assertGreater(got['below_bound'], 0)
        self.assertEqual(got['below_max'], 1024)
        self.assertTrue(any(line.startswith('CHUNK ROWS %s' % CMP) and '"partial": true' in line for line in lines))

    def test_the_stopped_arm_ends_at_the_death_line_not_past_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            with_death = self.crashed(tmp)[2]['chunk_rows'][CMP]
        with tempfile.TemporaryDirectory() as tmp:
            without = self.crashed(tmp, death=False)[2]['chunk_rows'][CMP]
        self.assertEqual(without['past_bound'] - with_death['past_bound'], 1, 'the row after the death line is counted only without a death line')

    def test_an_arm_that_never_ran_is_unavailable_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            _word, lines, facts = self.crashed(tmp)
        got = facts['chunk_rows'][NY]
        self.assertEqual(got, dict(unavailable='no final record and no journal header'))
        self.assertNotIn('below_bound', got)
        self.assertTrue(any(line.startswith('CHUNK ROWS %s' % NY) and 'unavailable' in line for line in lines))

    def test_a_whole_arm_is_not_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture5(tmp)
            tb.healthy(fx)
            tb.healthy(fx, NY)
            facts = fx.done()[2]
        self.assertNotIn('partial', facts['chunk_rows'][CMP])
        self.assertGreater(facts['chunk_rows'][CMP]['below_bound'], 0)

    def test_a_partial_span_past_the_bound_over_1024_still_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture5(tmp)
            tb.healthy(fx, starts=(320, 500, 700))
            fx.lines.append(tb.memory_line(tb.ORIGIN + 900, 'open', 0, 9000, 97152, 99200))
            (fx.root / 'engine-stream.log').write_text(''.join(fx.lines))
            fx.write_turns(CMP)
            facts = rule.report(str(fx.root))[2]
        self.assertEqual(facts['chunk_rows'][CMP]['past_over_1024'], 1)
        self.assertIn('chunk-rows: 1 chunks past the bound over 1024 rows', facts['reasons'])


class ChunkRows(unittest.TestCase):
    def test_the_read_sees_each_side_of_the_bound_on_rank_0_and_inside_the_span(self):
        rows = [dict(point='open', rank=0, t=5, a=0, b=2048), dict(point='open', rank=0, t=5, a=2048, b=4048), dict(point='open', rank=0, t=6, a=52624, b=53648), dict(point='open', rank=0, t=6, a=53648, b=55696),
                dict(point='released', rank=0, t=6, a=53648, b=55696), dict(point='open', rank=1, t=6, a=53648, b=55696), dict(point='open', rank=0, t=50, a=53648, b=55696)]
        self.assertEqual(rule.chunk_rows(rows, 1, 10), dict(below_bound=2, below_over_1024=2, below_max=2048, past_bound=2, past_over_1024=1, past_max=2048))


class BoxBinding(unittest.TestCase):
    def test_the_table_is_bound_for_the_call_and_put_back(self):
        seen = []
        with mock.patch.object(box4, 'replay_arm', side_effect=lambda *a, **k: seen.append(box4.T) or {}):
            with box._bound():
                self.assertIs(box4.T, T)
        self.assertIs(box4.T, T4)
        self.assertEqual(box4.ARM_SCHEMA, 'compose4-arm/1')

    def test_a_raise_puts_it_back_too(self):
        with self.assertRaises(ValueError):
            with box._bound():
                raise ValueError('x')
        self.assertIs(box4.T, T4)

    def test_the_argparse_takes_this_tables_selections_and_refuses_the_old_control(self):
        with mock.patch.object(box4, 'run_arm', return_value=0) as arm, tempfile.TemporaryDirectory() as tmp:
            args = ['arm', '--url', 'u', '--model', 'm', '--run-id', 'r', '--epoch', '1', '--c1', 'n22', '--size', '2', '--cohort', 'live', '--index', '1', '--workload', 'w', '--out', tmp + '/x.json']
            self.assertEqual(box.main(args), 0)
            self.assertEqual((arm.call_args[0][0].c1, arm.call_args[0][0].index), ('n22', 1))
            with self.assertRaises(SystemExit):
                box.main(args[:9] + ['cmp'] + args[10:])

    def test_the_smoke_goes_the_live_path_and_a_replay_the_replay_path(self):
        import types

        def go(c1, index):
            a = types.SimpleNamespace(c1=c1, size=2, cohort='live', index=index, url='u', run_id='r', epoch=1, control_timeout_s=1.0, out=str(Path(tmp) / ('%s-%d.json' % (c1, index))))
            with mock.patch.object(box4.refusal, 'RefusalKeepingClient'), mock.patch.object(box4.a1, 'load_config', return_value=({}, 1)), \
                    mock.patch.object(box4.a1, 'arm', return_value=dict(label='live', complete=True)) as live, mock.patch.object(box4, 'replay_arm', return_value=dict(label='replay', complete=True)) as play:
                self.assertEqual(box.run_arm(a), 0)
            return (live.call_args or play.call_args)[0][1], bool(live.called), bool(play.called)
        with tempfile.TemporaryDirectory() as tmp:
            row, live, play = go('sm', 0)
            self.assertEqual((row['label'], row['yield_on'], row['index'], live, play), (SM, True, 0, True, False))
            row, live, play = go('ny', 0)
            self.assertEqual((row['label'], row['yield_on'], row['ple_private'], live, play), (NY, False, False, False, True))

    def test_the_grade_names_this_run_and_reads_every_arm(self):
        grade = Grade()
        result = grade.grade_of(grade.records())
        self.assertEqual((result['verdict'], result['failures'], result['missing'], result['schema']), ('PASS', [], [], 'compose5-grade/1'))
        self.assertEqual(sorted(result['arms']), sorted(T.labels()))
        with box._bound():
            text = box.report(result)
        self.assertTrue(text.startswith('# q18a-compose-dnr'))
        self.assertIn('compose5_rule.py', text)


class Grade(unittest.TestCase):
    def records(self):
        out = []
        for row in T.arms():
            rec = dict(ta1.record_of(dict(row)), schema=box.ARM_SCHEMA, kind=row['kind'], label=row['label'])
            rec['inwait'] = dict(rec.get('inwait') or {}, bound_requested=row['chunk_bound'], bound_acknowledged=row['chunk_bound'])
            out.append(rec)
        return ta1.tbox.decode_records() + out

    def grade_of(self, arms, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'engine.log'
            log.write_text(''.join('INFO x %s\n' % line for line in tam.STARTUP))
            known = [a for a in arms if a.get('label')]
            return box.grade(arms, {a['label']: tap.PASS for a in known}, tap.CHECKS, [str(log)], {a['label']: tap.ALIVE for a in known}, **kw)

    def test_a_bound_the_engine_did_not_acknowledge_fails_on_every_arm(self):
        for row in T.arms():
            for reply in (None, [0, 1024], [52624, 2048]):
                arms = self.records()
                next(a for a in arms if a.get('label') == row['label'])['inwait']['bound_acknowledged'] = reply
                result = self.grade_of(arms)
                self.assertEqual(result['verdict'], 'FAIL', (row['label'], reply))
                self.assertIn(row['label'] + ':chunk-bound-not-acknowledged', result['failures'])

    def test_a_cap_a_rank_says_is_selected_fails_that_arm(self):
        arms = self.records()
        next(a for a in arms if a.get('label') == N22)['reserved_cap'] = dict(requested=False, select={str(k): dict(after=dict(reserved_cap=dict(cap_bytes=1, selected=True))) for k in range(4)})
        self.assertIn(N22 + ':reserved-cap-selected', self.grade_of(arms)['failures'])
        self.assertNotIn(NY + ':reserved-cap-selected', self.grade_of(arms)['failures'])

    def test_a_missing_arm_is_inc(self):
        arms = [a for a in self.records() if a['label'] != NY1]
        result = self.grade_of(arms)
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn(NY1 + ':no-arm-record', result['missing'])

    def test_either_off_arm_that_yielded_fails(self):
        for label in (NY, N22, N221, NY1):
            with self.subTest(label):
                arms = self.records()
                off = next(a for a in arms if a['label'] == label)
                off['statuses'][-1]['append_yield']['counters'] = dict(selected=9)
                off['statuses'][-1]['label'] = 'after'
                self.assertIn(label + ':yield-off-selected-a-yield', self.grade_of(arms)['failures'])


class Reads(UnderThisTable):
    def fixture(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Fixture5(tmp.name)

    def healthy(self, fx=None):
        fx = fx or self.fixture()
        tb.healthy(fx)
        tb.healthy(fx, NY)
        return fx

    def test_the_price_over_known_numbers(self):
        fx = self.healthy()
        fx.write_record(PAIRS[0][0], **frozen(bulk=23.0, decode=57.12))
        fx.write_record(PAIRS[0][1], **frozen(bulk=21.0, decode=56.0, selected=0))
        fx.write_record(PAIRS[1][0], **frozen(bulk=23.5, decode=56.0))
        fx.write_record(PAIRS[1][1], **frozen(bulk=21.5, decode=56.0, selected=0))
        _word, lines, facts = fx.done()
        price = facts['price']
        self.assertEqual([p['bulk_extra_s'] for p in price['pairs']], [2.0, 2.0])
        self.assertEqual([p['bulk_on_over_off'] for p in price['pairs']], [round(23 / 21, 4), round(23.5 / 21.5, 4)])
        self.assertEqual(price['pairs'][0]['decode_per_stream_on_over_off'], 1.02)
        self.assertEqual((price['on_on_bulk_spread_s'], price['off_off_bulk_spread_s']), (0.5, 0.5))
        self.assertEqual(price['bulk_ratio_spread'], round(abs(round(23 / 21, 4) - round(23.5 / 21.5, 4)), 4))
        self.assertEqual(sum(1 for line in lines if line.startswith('PRICE fon')), 2)
        self.assertIn('PRICE NOISE', '\n'.join(lines))

    def test_the_price_is_never_a_word(self):
        calm = self.healthy()
        word, _l, _f = calm.done()
        dear = self.healthy()
        dear.write_record(PAIRS[0][0], **frozen(bulk=90.0))
        dear.write_record(PAIRS[1][0], **frozen(bulk=90.0))
        self.assertEqual(dear.done()[0], word)

    def test_a_frozen_arm_with_no_record_reads_as_none_and_the_rule_says_inc_or_worse(self):
        fx = self.healthy()
        path = fx.root / ('arm-%s.json' % PAIRS[1][1])
        path.rename(path.with_suffix('.gone'))
        (fx.root / ('arm-%s.log' % PAIRS[1][1])).rename(fx.root / 'gone.log')
        word, lines, facts = fx.done()
        self.assertEqual(word, 'INC', lines[-3:])
        self.assertIsNone(facts['price']['pairs'][1]['off_read'])
        self.assertIn('not-run: ' + PAIRS[1][1], facts['reasons'])

    def test_r1_is_read_on_the_smoke_and_on_the_pair_and_is_not_in_the_verdict(self):
        fx = self.healthy()
        fx.write_record(SM, wall_start_s=tb.ORIGIN - 400, wall_end_s=tb.ORIGIN - 340, returners=[dict(suffix_tokens=500, ttft_s=1.5)], score_warm=dict(counters=dict(rewarms={'1': 7}, allocs={'1': 8})), reserved_cap=dict(counters=dict(overdrafts={'1': 3}, overdraft_refused={'1': 0})))
        word, lines, facts = fx.done()
        self.assertEqual((facts['r1'][SM]['rewarms'], facts['r1'][SM]['allocs'], facts['r1'][SM]['overdrafts']), (7, 8, 3))
        self.assertIn('SMOKE R1', '\n'.join(lines))
        clean = self.healthy().done()[0]
        self.assertEqual(word, clean, 'a read, not a verdict')

    def test_a_stock_rescue_on_cmp_is_inc_and_on_ny_is_nothing(self):
        fx = self.healthy()
        clean = fx.done()[0]
        self.assertEqual(clean, 'PASS')
        rescued = dict(stock_rescue=dict(counters=dict(attempts={'1': 2}, refused={'1': 0})))
        self.assertEqual(rule.stock_rescues(dict(record=rescued)), 2)
        self.assertEqual(rule.stock_rescues(dict(record={})), 0)
        path = fx.root / ('arm-%s.json' % CMP)
        record = json.loads(path.read_text())
        record.update(rescued)
        path.write_text(json.dumps(record))
        word, lines, facts = rule.report(str(fx.root))
        self.assertEqual((word, facts['stock_rescues'], lines[-1]), ('INC', 2, 'RULE INC'))
        self.assertIn('REASON stock-rescue: cmp 2', lines)

    def test_no_line_names_the_old_control_and_compose_4s_rule_is_put_back(self):
        _word, lines, _facts = self.healthy().done()
        self.assertFalse([line for line in lines if ' ctl' in line or 'ctl ' in line], lines)
        self.assertEqual((compose4_rule.OFF, compose4_rule.ARMS[-1]), ('ctl.g2-live-0', 'ctl.g2-live-0'))

    def test_it_runs_as_the_lead_runs_it_with_the_standard_library_alone(self):
        fx = self.healthy()
        fx.done()
        proc = subprocess.run([sys.executable, '-I', rule.__file__, str(fx.root)], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:] + proc.stdout[-800:])
        self.assertIn('RULE PASS', proc.stdout)
        source = Path(rule.__file__).read_text()
        self.assertNotRegex(source, r'(?m)^(import|from) (compose5_arms|e3_|torch|numpy)')


if __name__ == '__main__':
    unittest.main()
