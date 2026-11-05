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
SM, DEC1, CMP, NY = rule.SM, 'dec1.g2-live-0', rule.ON, rule.OFF
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
        self.assertEqual(T.labels(), [SM, DEC1, 'fon.g2-live-0', 'foff.g2-live-0', 'foff.g2-live-1', 'fon.g2-live-1', CMP, NY])
        self.assertEqual(tuple(T.labels()), rule.ARMS)
        self.assertEqual(tuple(T.price_pairs()), PAIRS)

    def test_each_pair_differs_in_the_yield_alone(self):
        rows = T.by_label()
        for on, off in ((CMP, NY),) + PAIRS:
            self.assertEqual(sorted(k for k in rows[on] if rows[on][k] != rows[off][k]), ['c1', 'label', 'selection', 'yield_on'], (on, off))
            self.assertEqual((rows[on]['yield_on'], rows[off]['yield_on']), (True, False))

    def test_the_private_ple_buffer_is_on_every_arm_and_the_witness_on_none(self):
        for row in T.arms():
            self.assertEqual((row['ple_private'], row['ple_witness']), (True, False), row['label'])

    def test_compose_4s_rows_are_the_base(self):
        c4 = T4.by_label()[CMP]
        row = T.by_label()[CMP]
        self.assertEqual({k: row[k] for k in row if k not in c4 or row[k] != c4[k]}, dict(chunk_bound=T.B13_BOUND), 'cmp is compose-4\'s cmp and b13')
        self.assertEqual(sorted(k for k in c4 if c4[k] != T.by_label()[NY][k]), ['c1', 'label', 'selection', 'yield_on'], 'ny is cmp without the yield')

    def test_the_replay_and_the_ceiling_are_compose_4s(self):
        self.assertIs(T.REPLAY, T4.REPLAY)
        self.assertEqual(T.MAX_PROMPT_TOKENS, 135000)
        self.assertEqual((T.RULE_PREFILL_TOK_S, T.RULE_DECODE_VS_FIXED, T.RULE_MIN_NATIVE_BULKS), (T4.RULE_PREFILL_TOK_S, T4.RULE_DECODE_VS_FIXED, T4.RULE_MIN_NATIVE_BULKS))

    def test_a_live_arms_request_count_and_a_replays(self):
        self.assertEqual(T.by_label()['fon.g2-live-1']['generations'], 12)
        self.assertEqual(T.by_label()['foff.g2-live-1']['generations'], 12)
        self.assertIsNone(T.by_label()[NY]['generations'])

    def test_a_matched_off_only_for_the_on_arms(self):
        rows = T.by_label()
        self.assertEqual(T.matched_off(rows['fon.g2-live-1']), 'foff.g2-live-1')
        self.assertEqual(T.matched_off(rows[CMP]), NY)
        self.assertIsNone(T.matched_off(rows[SM]))


class Smoke(unittest.TestCase):
    """What this table adds to compose-4's smoke; b13h's numbers in the record are dp8d's (q18a-dp8-c32, arm-b13h.g2-live-0.json)."""

    def full(self):
        row = T.by_label()[SM]
        rec = ta1.record_of(row)
        rec.update(ple_private=dict(counters=dict(scopes={'0': 9, '1': 9})), score_warm=dict(counters=dict(allocs={str(k): 1 for k in range(4)})), reserved_cap=dict(requested=True, counters={}),
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
            args = ['arm', '--url', 'u', '--model', 'm', '--run-id', 'r', '--epoch', '1', '--c1', 'foff', '--size', '2', '--cohort', 'live', '--index', '1', '--workload', 'w', '--out', tmp + '/x.json']
            self.assertEqual(box.main(args), 0)
            self.assertEqual((arm.call_args[0][0].c1, arm.call_args[0][0].index), ('foff', 1))
            with self.assertRaises(SystemExit):
                box.main(args[:9] + ['ctl'] + args[10:])

    def test_a_frozen_arm_goes_the_live_path_and_a_replay_the_replay_path(self):
        import types

        def go(c1, index):
            a = types.SimpleNamespace(c1=c1, size=2, cohort='live', index=index, url='u', run_id='r', epoch=1, control_timeout_s=1.0, out=str(Path(tmp) / ('%s-%d.json' % (c1, index))))
            with mock.patch.object(box4.refusal, 'RefusalKeepingClient'), mock.patch.object(box4.a1, 'load_config', return_value=({}, 1)), \
                    mock.patch.object(box4.a1, 'arm', return_value=dict(label='live', complete=True)) as live, mock.patch.object(box4, 'replay_arm', return_value=dict(label='replay', complete=True)) as play:
                self.assertEqual(box.run_arm(a), 0)
            return (live.call_args or play.call_args)[0][1], bool(live.called), bool(play.called)
        with tempfile.TemporaryDirectory() as tmp:
            row, live, play = go('foff', 1)
            self.assertEqual((row['label'], row['yield_on'], row['index'], live, play), ('foff.g2-live-1', False, 1, True, False))
            row, live, play = go('ny', 0)
            self.assertEqual((row['label'], row['yield_on'], row['ple_private'], live, play), (NY, False, True, False, True))

    def test_the_grade_names_this_run_and_reads_every_arm(self):
        grade = Grade()
        result = grade.grade_of(grade.records())
        self.assertEqual((result['verdict'], result['failures'], result['missing'], result['schema']), ('PASS', [], [], 'compose5-grade/1'))
        self.assertEqual(sorted(result['arms']), sorted(T.labels()))
        with box._bound():
            text = box.report(result)
        self.assertTrue(text.startswith('# q18a-compose-5'))
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

    def test_a_missing_frozen_arm_is_inc(self):
        arms = [a for a in self.records() if a['label'] != 'foff.g2-live-1']
        result = self.grade_of(arms)
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('foff.g2-live-1:no-arm-record', result['missing'])

    def test_either_off_arm_that_yielded_fails(self):
        for label in (NY, 'foff.g2-live-0', 'foff.g2-live-1'):
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
