#!/usr/bin/env python3
"""q18a-compose-4: the table, the replay client against a scripted stream, the arm's control order, the smoke and the grade over synthetic records, and the rule both ways on fixture records. CPU only; no pass here is box evidence. The arm through the real router: test_compose4_router."""
import contextlib
import datetime
import io
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest import mock

import test_a1_trunk_box as ta1   # first: puts tools/ and the engine sources on the path
import compose4_arms as T
import compose4_replay as replay
import compose4_rule as rule
import e3_compose4_box as box

tam, tap = ta1.tam, ta1.tap
CMP, CTL, DEC1, SM = 'cmp.g2-live-0', 'ctl.g2-live-0', 'dec1.g2-live-0', 'sm.g2-live-0'
ORIGIN = 1_800_000_000.0


class Table(unittest.TestCase):
    def test_the_order_and_the_one_axis(self):
        self.assertEqual(T.labels(), [SM, DEC1, CMP, CTL])
        rows = T.by_label()
        self.assertEqual(sorted(k for k in rows[CMP] if rows[CMP][k] != rows[CTL][k]), ['c1', 'label', 'ple_private', 'selection', 'yield_on'])
        for r in T.arms():
            for name in ('score_warm', 'reserved_cap', 'exact_successor', 'allocator_flush', 'cow_fast', 'cow_host_filter', 'cow_verify', 'inwait', 'next_levers', 'prepare_slices'):
                self.assertIs(r[name], True, (r['label'], name))
            self.assertIs(r['ple_witness'], False, r['label'])

    def test_the_rule_reads_the_tables_numbers(self):
        self.assertEqual((rule.RULE_PREFILL_TOK_S, rule.RULE_DECODE_VS_FIXED, rule.RULE_MIN_NATIVE_BULKS, rule.FIXED_REFERENCE_TOK_S), (T.RULE_PREFILL_TOK_S, T.RULE_DECODE_VS_FIXED, T.RULE_MIN_NATIVE_BULKS, T.FIXED_REFERENCE_TOK_S))
        self.assertEqual((rule.MIN_NATIVE_ROWS, rule.MAX_NEW, rule.PER_BULK), (T.MIN_NATIVE_ROWS, T.LAUNCH['max_new'], T.LAUNCH['per_bulk']))
        self.assertEqual(tuple(T.labels()), rule.ARMS)

    def test_the_rule_runs_with_the_standard_library_alone(self):
        source = Path(rule.__file__).read_text()
        self.assertNotRegex(source, r'(?m)^(import|from) (compose4|e3_|torch|numpy)')


class Clock:
    def __init__(self):
        self.now, self.lock = 0.0, threading.Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def tick(self, seconds):
        with self.lock:
            self.now += seconds


def scripted(clock, prompts):
    def stream(_url, body, _key, origin, **_):
        words = sum(len(m['content'].split()) for m in body['messages'])
        tokens = int(words * 2.2)
        with threading.Lock():
            prompts.append((body['cache_salt'], tokens))
        clock.tick(1.0)
        reply = body['max_tokens']
        return dict(ttft=0.5, seconds=1.0, text='ok', finish='length', arrivals_ms=[0, 1], completion_tokens=reply, prompt_tokens=tokens, cached_tokens=0)
    return stream


class Replay(unittest.TestCase):
    CFG = dict(T.REPLAY, AGENTS=1, DURATION=400.0, WARMUP=0.0, START_MIN=120000, START_MAX=120000)

    def play(self, cfg=None):
        clock, prompts = Clock(), []
        stream = scripted(clock, prompts)
        got = replay.run('http://x', 'm', 'k', 'run', CMP, cfg=cfg or self.CFG, clock=clock, sleep=clock.tick, stream=stream)
        return got, prompts

    def test_no_prompt_passes_the_ceiling_and_an_append_over_it_rolls_the_session(self):
        (_cfg, _summary, turns, still_open), prompts = self.play()
        self.assertEqual(still_open, [])
        self.assertGreater(len({t['session'] for t in turns}), 1, 'the ceiling rolled a session')
        self.assertLessEqual(max(tokens for _salt, tokens in prompts), T.MAX_PROMPT_TOKENS)
        self.assertGreater(sum(1 for t in turns if t['kind'] == 'append'), 2)

    def test_two_arms_send_different_salts_the_same_draws(self):
        (_c, _s, one, _o), first = self.play()
        clock, prompts = Clock(), []
        replay.run('http://x', 'm', 'k', 'run', CTL, cfg=self.CFG, clock=clock, sleep=clock.tick, stream=scripted(clock, prompts))
        self.assertFalse({s for s, _t in first} & {s for s, _t in prompts}, 'no arm reads another arm\'s cache')
        self.assertEqual([t for _s, t in first], [t for _s, t in prompts], 'seeded: the same prompts')
        self.assertTrue(one)

    def test_a_failed_turn_starts_a_new_session(self):
        clock = Clock()
        calls = []

        def stream(_url, body, _key, _origin, **_):
            clock.tick(1.0)
            calls.append(body)
            if len(calls) == 2:
                raise TimeoutError('timed out')
            return dict(ttft=0.5, seconds=1.0, text='ok', finish='length', arrivals_ms=[0, 1], completion_tokens=body['max_tokens'], prompt_tokens=60000, cached_tokens=0)
        _c, _s, turns, _o = replay.run('http://x', 'm', 'k', 'run', CMP, cfg=dict(self.CFG, DURATION=30.0), clock=clock, sleep=clock.tick, stream=stream)
        self.assertIn('error', turns[1])
        self.assertEqual(turns[2]['kind'], 'fresh')

    def test_the_key_comes_from_the_environment_and_never_into_the_record(self):
        with mock.patch.dict('os.environ', {'VLLM_API_KEY': 'k-xyz'}):
            self.assertEqual(replay.key_from_env(), 'k-xyz')
        (cfg, summary, turns, _o), _p = self.play()
        self.assertNotIn('k-xyz', json.dumps([cfg, summary, turns]))


def stamp(at):
    return datetime.datetime.fromtimestamp(at, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z'


def memory_line(at, point, rank, txn, a, b, *, free=2048, retries=0, ooms=0):
    return '%s INFO flash-s5 memory point=%s rank=%d txn_seq=%d chunk=%d-%d allocated=1 reserved=1 inactive_split=0 alloc_retries=%d ooms=%d free=%d total=9\n' % (stamp(at), point, rank, txn, a, b, retries, ooms, free * 1048576)


class Fixture:
    """A record dir: engine lines, arm records, turns. Offsets in seconds from a replay's origin."""

    def __init__(self, tmp):
        self.root, self.lines, self.txn, self.turns, self.origin = Path(tmp), [], 0, {}, {}
        self.cap = {}
        for label, at in ((SM, ORIGIN - 400), (DEC1, ORIGIN - 300)):
            self.write_record(label, wall_start_s=at, wall_end_s=at + 60, reasons=[], complete=True, returners=[dict(suffix_tokens=500, ttft_s=1.5)], decode_only=dict(aggregate_tokens_per_s=60.0, stream_rates_tokens_per_s=[60.0], streams=[dict(gaps_ms=[16.0, 17.0, 40.0])]))
        (self.root / 'arm-decode-am1.json').write_text(json.dumps(dict(decode_only=dict(aggregate_tokens_per_s=456.0, stream_rates_tokens_per_s=[57.0] * 8, streams=[dict(gaps_ms=[17.0, 18.0, 60.0])]))))
        (self.root / 'engine-death-scan.json').write_text(json.dumps(dict(verdict='ALIVE')))
        for label, origin in ((CMP, ORIGIN), (CTL, ORIGIN + 2000)):
            self.origin[label] = origin
            self.turns[label] = []

    def write_record(self, label, **fields):
        (self.root / ('arm-%s.log' % label)).write_text('')
        (self.root / ('arm-%s.json' % label)).write_text(json.dumps(dict(dict(label=label, complete=True, reasons=[]), **fields)))

    def native(self, label, start, end, a, b, *, seconds_per_chunk=0.3, rows=1024, free=2048):
        """A native run on rank 0, one chunk per ``rows``, from ``start``; returns the time of its last release."""
        at = self.origin[label] + start
        while a < b:
            following = min(a + rows, b)
            self.txn += 1
            self.lines.append(memory_line(at, 'open', 0, self.txn, a, following, free=free))
            at += seconds_per_chunk
            self.lines.append(memory_line(at, 'released', 0, self.txn, a, following, free=free))
            a = following
        assert end is None or at <= self.origin[label] + end + 1
        return at - self.origin[label]

    def turn(self, label, kind, t, *, prompt, cached=0, ttft=1.0, reply=256, agent=0, gap_ms=18.0, **more):
        first = (t + ttft) * 1000.0
        arrivals = [int(first + k * gap_ms) for k in range(reply)]
        row = dict(agent=agent, session=1, kind=kind, t=t, history_est=prompt, reply_target=reply, ttft=ttft, seconds=ttft + reply * gap_ms / 1000.0, completion_tokens=reply, prompt_tokens=prompt, cached_tokens=cached, finish='length', arrivals_ms=arrivals, complete=True)
        row.update(more)
        self.turns[label].append(row)
        return row

    def fresh_native(self, label, t, *, prompt=65536, agent=0, ttft=None, seconds_per_chunk=0.2):
        last = self.native(label, t + 0.2, None, 0, prompt - 1, seconds_per_chunk=seconds_per_chunk)
        return self.turn(label, 'fresh', t, prompt=prompt, ttft=ttft or (last - t + 0.2), agent=agent)

    def done(self, *, cap=None):
        (self.root / 'engine-stream.log').write_text(''.join(self.lines))
        for label in (CMP, CTL):
            origin = self.origin[label]
            self.write_record(label, wall_start_s=origin - 5, wall_end_s=origin + 1090, replay_start_s=origin, replay_config=dict(T.REPLAY), replay_summary=dict(warm_turns=len(self.turns[label])), replay_still_open=[], reserved_cap=dict(counters=(cap or {}).get(label, dict(overdrafts={'0': 0}, overdraft_refused={'0': 0}))))
            (self.root / ('arm-%s.json.turns.json' % label)).write_text(json.dumps(self.turns[label]))
        return rule.report(str(self.root))


def healthy(fx, label=CMP, *, starts=(320, 500, 700, 900)):
    """Native bulks at 3500 tok/s on the client's basis, decoders beside them at 50 tok/s."""
    for k, t in enumerate(starts):
        fx.fresh_native(label, t, agent=k, ttft=65536 / 3500.0)
        for other in range(1, 4):
            fx.turn(label, 'append', t + 1.0 * other + 25, prompt=70000, cached=67000, ttft=0.8, agent=other + 10, gap_ms=20.0)
    # a decoder through each bulk, sent before the bulk: its reply covers the bulk's span
    for t in starts:
        fx.turn(label, 'append', t - 1.5, prompt=70000, cached=67000, ttft=0.2, reply=800, agent=7, gap_ms=20.0)


class Rule(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fx = Fixture(self.tmp.name)

    def test_a_clean_run_passes_and_the_report_leads_with_the_native_share(self):
        healthy(self.fx)
        healthy(self.fx, CTL)
        word, lines, facts = self.fx.done()
        self.assertEqual(word, 'PASS', lines)
        self.assertTrue(lines[0].startswith('NATIVE SHARE'), lines[0])
        self.assertEqual(lines[-1], 'RULE PASS')
        read = facts['arms'][CMP]['read']
        self.assertEqual(read['native_bulks'], 4)
        self.assertGreater(read['native_share'], 0.5)
        self.assertGreater(read['prefill_tok_s'], 3300)

    def test_the_review_reads_are_there_per_arm_and_the_debt_is_apart_from_the_prefill(self):
        healthy(self.fx)
        healthy(self.fx, CTL)
        _word, lines, facts = self.fx.done()
        text = '\n'.join(lines)
        for name in ('RATE RETENTION', 'TAIL GAPS ms', 'RETURN-TURN FIRST OUTPUT s', 'COMPLETED VALID TURNS', 'DECODE DEBT', 'PREFILL DURATION s, apart from the debt'):
            self.assertIn(name, text)
        for label in (CMP, CTL):
            serving = facts['arms'][label]['read']['serving']
            self.assertEqual(serving['completed_valid_turns']['not_valid'], 0)
            self.assertGreater(serving['tail_gaps_ms']['beside_native_bulks']['n'], 0)
            self.assertEqual(serving['tail_gaps_ms']['window']['p99'], 20)
            debt = serving['decode_debt']
            # 20 ms a token is 50 tok/s against 56.6: the debt is the difference over the stream-seconds, whatever the bulks took.
            self.assertAlmostEqual(debt['debt_share'], 1 - 50.0 / 56.6, delta=0.03)
            self.assertAlmostEqual(serving['rate_retention']['beside_native_bulks'], 50.0 / 56.6, delta=0.03)
            self.assertAlmostEqual(serving['prefill_duration_s']['fresh_native'], 4 * 65536 / 3500.0, delta=1.0)

    def test_coalesced_pieces_count_as_their_tokens(self):
        turn = dict(arrivals_ms=[0, 40, 80], completion_tokens=6)
        self.assertEqual(rule.tokens_per_arrival(turn), 2.0)
        self.assertEqual(rule.tokens_per_arrival(dict(arrivals_ms=[], completion_tokens=6)), 1.0)

    def test_a_clean_fresh_that_went_stock_fails_clause_a(self):
        healthy(self.fx)
        healthy(self.fx, CTL)
        self.fx.turn(CMP, 'append', 1000, prompt=70000, cached=67000, ttft=0.2, reply=800, agent=3, gap_ms=20.0)
        self.fx.turn(CMP, 'fresh', 1003, prompt=90000, cached=0, ttft=20.0, agent=4)
        word, lines, facts = self.fx.done()
        self.assertEqual(word, 'FAIL', lines)
        self.assertEqual(facts['arms'][CMP]['read']['violations'][0]['clause'], 'a')
        self.assertIn('stock-prefill (a)', ' '.join(facts['reasons']))

    def test_a_contended_fresh_or_one_with_no_decoder_is_counted_and_does_not_fail(self):
        healthy(self.fx)
        healthy(self.fx, CTL)
        self.fx.turn(CMP, 'fresh', 1095, prompt=90000, cached=0, ttft=20.0, agent=4, reply=64)     # nothing decoding at 1095: past the window's end for the others, inside for it
        self.fx.turn(CMP, 'append', 1000, prompt=70000, cached=67000, ttft=0.2, reply=4000, agent=3, gap_ms=20.0)    # decoding until 1080
        self.fx.turn(CMP, 'fresh', 1020, prompt=90000, cached=0, ttft=25.0, agent=5, reply=64)
        self.fx.turn(CMP, 'fresh', 1020.2, prompt=90000, cached=0, ttft=30.0, agent=6, reply=64)
        word, lines, facts = self.fx.done()
        counts = facts['arms'][CMP]['read']['counts']
        self.assertEqual(word, 'PASS', lines)
        self.assertEqual(counts.get('stock_fresh_contended'), 2)
        self.assertEqual(counts.get('stock_fresh_no_decoder'), 1)
        self.assertIn('LITERAL reading', '\n'.join(lines))

    def test_a_small_append_stock_inside_a_live_bulk_fails_clause_b_on_the_yield_arm_only(self):
        for label in (CMP, CTL):
            healthy(self.fx, label)
            self.fx.fresh_native(label, 1000, agent=20, ttft=65536 / 3500.0, seconds_per_chunk=0.5)
            self.fx.turn(label, 'append', 1005, prompt=70000, cached=68500, ttft=4.0, agent=21)
        word, lines, facts = self.fx.done()
        self.assertEqual(word, 'FAIL', lines)
        self.assertEqual([v['clause'] for v in facts['arms'][CMP]['read']['violations']], ['b'])
        self.assertEqual(facts['arms'][CTL]['read']['violations'], [])
        self.assertEqual(facts['arms'][CTL]['read']['counts'].get('stock_append_yield_off'), 1)

    def test_an_append_over_max_new_or_late_in_the_bulk_or_with_the_yields_spent_is_counted_only(self):
        healthy(self.fx)
        healthy(self.fx, CTL)
        self.fx.fresh_native(CMP, 1000, agent=20, ttft=65536 / 3500.0, seconds_per_chunk=0.5)
        self.fx.turn(CMP, 'append', 1005, prompt=70000, cached=67000, ttft=4.0, agent=21)    # 3000 new
        self.fx.turn(CMP, 'append', 1031.5, prompt=70000, cached=68500, ttft=4.0, agent=22)  # under 2 s left
        word, lines, facts = self.fx.done()
        counts = facts['arms'][CMP]['read']['counts']
        self.assertEqual(word, 'PASS', lines)
        self.assertEqual((counts.get('stock_append_over_max_new'), counts.get('stock_append_bulk_ending')), (1, 1))

    def test_a_served_append_counts_as_native_with_its_first_output(self):
        healthy(self.fx)
        healthy(self.fx, CTL)
        last = self.fx.native(CMP, 1000.2, None, 67000, 68999)
        self.fx.turn(CMP, 'append', 1000, prompt=69000, cached=0, ttft=last - 1000 + 0.2, agent=21)
        _word, _lines, facts = self.fx.done()
        read = facts['arms'][CMP]['read']
        self.assertEqual(read['counts']['append_served'], 1)
        self.assertEqual(read['returner_first_output_s']['all']['n'], 1)

    def test_fatal_oom_and_a_dead_engine_and_open_turns_and_a_read_timeout_each_fail(self):
        healthy(self.fx)
        healthy(self.fx, CTL)
        self.fx.lines.append(stamp(ORIGIN) + ' ERROR torch.OutOfMemoryError: CUDA out of memory\n')
        word, _lines, facts = self.fx.done()
        self.assertEqual(word, 'FAIL')
        self.assertIn('OutOfMemoryError', ' '.join(facts['reasons']))
        self.fx.lines.pop()
        (self.fx.root / 'engine-death-scan.json').write_text(json.dumps(dict(verdict='DEAD')))
        self.assertEqual(self.fx.done()[0], 'FAIL')
        (self.fx.root / 'engine-death-scan.json').write_text(json.dumps(dict(verdict='ALIVE')))
        self.fx.turns[CTL].append(dict(agent=1, session=1, kind='append', t=1000.0, error='TimeoutError:timed out'))
        word, _lines, facts = self.fx.done()
        self.assertEqual(word, 'FAIL')
        self.assertIn('read timed out', ' '.join(facts['reasons']))

    def test_a_rescue_on_cmp_is_inc_and_on_ctl_only_a_read(self):
        healthy(self.fx)
        healthy(self.fx, CTL)
        word, lines, facts = self.fx.done(cap={CMP: dict(overdrafts={'1': 3}, overdraft_refused={'1': 0})})
        self.assertEqual(word, 'INC', lines)
        self.assertIn('rescued: cmp 3', ' '.join(facts['reasons']))
        word, lines, _facts = self.fx.done(cap={CTL: dict(overdrafts={'1': 3}, overdraft_refused={'1': 0})})
        self.assertEqual(word, 'PASS', lines)
        self.assertIn("'1': 3", '\n'.join(lines).replace('"', "'"))

    def test_under_three_bulks_or_under_the_floors_is_inc(self):
        healthy(self.fx, starts=(320, 500))
        healthy(self.fx, CTL)
        word, _l, facts = self.fx.done()
        self.assertEqual(word, 'INC')
        self.assertIn('too-few', ' '.join(facts['reasons']))

    def test_a_slow_median_is_inc(self):
        for t in (320, 500, 700):
            self.fx.fresh_native(CMP, t, ttft=65536 / 3000.0)
            self.fx.turn(CMP, 'append', t - 1.5, prompt=70000, cached=67000, ttft=0.2, reply=800, agent=7, gap_ms=20.0)
        healthy(self.fx, CTL)
        word, _l, facts = self.fx.done()
        self.assertEqual(word, 'INC')
        self.assertIn('prefill: cmp', ' '.join(facts['reasons']))

    def test_slow_decode_is_inc(self):
        for t in (320, 500, 700):
            self.fx.fresh_native(CMP, t, ttft=65536 / 3500.0)
            self.fx.turn(CMP, 'append', t - 1.5, prompt=70000, cached=67000, ttft=0.2, reply=800, agent=7, gap_ms=40.0)
        healthy(self.fx, CTL)
        word, _l, facts = self.fx.done()
        self.assertEqual(word, 'INC')
        self.assertIn('decode: cmp', ' '.join(facts['reasons']))

    def test_a_queued_bulk_is_out_of_the_median_but_counts_as_a_bulk(self):
        healthy(self.fx)
        healthy(self.fx, CTL)
        last = self.fx.native(CMP, 1005, None, 0, 65535)
        self.fx.turn(CMP, 'fresh', 1000, prompt=65536, ttft=last - 1000 + 0.2, agent=30)
        _w, _l, facts = self.fx.done()
        read = facts['arms'][CMP]['read']
        self.assertEqual((read['native_bulks'], read['prefill_n']), (5, 4))

    def test_a_turn_sent_in_the_warmup_is_outside_the_window(self):
        healthy(self.fx)
        healthy(self.fx, CTL)
        self.fx.turn(CMP, 'append', 100, prompt=70000, cached=67000, ttft=0.2, reply=800, agent=3)
        self.fx.turn(CMP, 'fresh', 105, prompt=90000, cached=0, ttft=20.0, agent=4)
        self.assertEqual(self.fx.done()[0], 'PASS')

    def test_chunks_that_continue_a_run_are_one_bulk_across_a_yield(self):
        self.fx.native(CMP, 320, None, 0, 4096)
        self.fx.native(CMP, 325, None, 67000, 69000)
        self.fx.native(CMP, 330, None, 4096, 8192)
        path = self.fx.root / 'lines.log'
        path.write_text(''.join(self.fx.lines))
        rows, _fatal = rule.read_log(str(path))
        runs = rule.chunk_runs(rows)
        self.assertEqual(sorted((r['start'], r['end']) for r in runs), [(0, 8192), (67000, 69000)])

    def test_memory_reads_are_per_rank_and_only_the_arms_span(self):
        rows = [dict(t=10.0, rank=1, point='open', txn=1, a=0, b=1, free=500, retries=0, ooms=2), dict(t=11.0, rank=1, point='released', txn=1, a=0, b=1, free=400, retries=2, ooms=3), dict(t=99.0, rank=1, point='open', txn=2, a=1, b=2, free=1, retries=9, ooms=9)]
        got = rule.arm_memory(rows, 5.0, 20.0)
        self.assertEqual((got['min_free_mib'], got['alloc_retries'], got['oom_rise']), ({'1': 400}, {'1': 2}, {'1': 1}))

    def test_the_memory_line_format_is_the_engines(self):
        source = (Path(__file__).resolve().parent.parent / 'patches/s5_engine').glob('*.py')
        text = ''.join(p.read_text() for p in source)
        self.assertIn('flash-s5 memory point=', text)
        for field in ('rank=', 'txn_seq=', 'chunk=', 'alloc_retries=', 'ooms=', 'free='):
            self.assertIn(field, text)

    def test_the_cli_returns_its_verdict_and_never_overwrites(self):
        healthy(self.fx)
        healthy(self.fx, CTL)
        self.fx.done()
        out = Path(self.tmp.name) / 'facts.json'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rule.main([self.tmp.name, '--json', str(out)]), 0)
        self.assertEqual(json.loads(out.read_text())['rule'], 'PASS')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            rule.main([self.tmp.name, '--json', str(out)])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rule.main([]), 2)


class Grade(unittest.TestCase):
    def records(self):
        out = []
        for row in T.arms():
            out.append(dict(ta1.record_of(dict(row)), schema=box.ARM_SCHEMA, kind=row['kind'], label=row['label']))
        return ta1.tbox.decode_records() + out

    def grade_of(self, arms, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'engine.log'
            log.write_text(''.join('INFO x %s\n' % line for line in tam.STARTUP))
            known = [a for a in arms if a.get('label')]
            return box.grade(arms, {a['label']: tap.PASS for a in known}, tap.CHECKS, [str(log)], {a['label']: tap.ALIVE for a in known}, **kw)

    def test_a_whole_record_passes(self):
        result = self.grade_of(self.records())
        self.assertEqual((result['verdict'], result['failures'], result['missing']), ('PASS', [], []))
        self.assertTrue(box.report(result).startswith('# q18a-compose-4'))

    def test_a_missing_arm_is_inc_and_an_off_arm_that_yielded_fails(self):
        arms = self.records()
        self.assertEqual(self.grade_of(arms[:-1])['verdict'], 'INC')
        arms = self.records()
        off = next(a for a in arms if a['label'] == CTL)
        off['statuses'][-1]['append_yield']['counters'] = dict(selected=9)
        off['statuses'][-1]['label'] = 'after'
        self.assertIn(CTL + ':yield-off-selected-a-yield', self.grade_of(arms)['failures'])


class Smoke(unittest.TestCase):
    def full(self):
        rec = ta1.record_of(T.by_label()[SM])
        rec.update(ple_private=dict(counters=dict(scopes={'0': 9, '1': 9})), score_warm=dict(counters=dict(allocs={str(k): 1 for k in range(4)})), reserved_cap=dict(requested=True, counters={}))
        rec.setdefault('inwait', {})['exact_acknowledged'] = True
        return rec

    def test_a_missing_lever_stops_the_table(self):
        for name, change, reason in (('ple', lambda r: r.update(ple_private={}), 'smoke-ple-private-not-engaged'), ('warm', lambda r: r.update(score_warm={}), 'smoke-score-warm-counters-absent'), ('cap', lambda r: r.update(reserved_cap={}), 'smoke-reserved-cap-not-requested'),
                                     ('exact', lambda r: r['inwait'].update(exact_acknowledged=False), 'smoke-exact-successor-not-acknowledged')):
            with self.subTest(name):
                rec = self.full()
                change(rec)
                ok, reasons = box.smoke_check(rec)
                self.assertFalse(ok)
                self.assertIn(reason, reasons)


class ArmOrder(unittest.TestCase):
    """The replay arm's control around a scripted run: entry, the yield's selection, the release, the replay, the reads."""

    def status(self, **auto):
        return dict(route='full', scheduler_requests=0, route_generation=3, auto_arm=dict(dict(manual_owned=True, active=False), **auto), append_yield=dict(config=dict(T.LAUNCH), phase=None))

    def test_the_arm_refuses_a_gate_that_is_not_pass(self):
        a = types.SimpleNamespace(run_id='r', url='http://x', model='m')
        client = types.SimpleNamespace(epoch=1, history=[], call=lambda *_a, **_k: self.status())
        with mock.patch.object(box.shared, '_package_gate', return_value={'ok': False}):
            record, turns = box.replay_arm(a, T.by_label()[CMP], client, client, run=None)
        self.assertEqual((record['complete'], record['reasons'], turns), (False, ['arm-error'], None))

    def test_an_arm_entry_that_is_not_idle_full_is_refused(self):
        client = types.SimpleNamespace(call=lambda *_a, **_k: dict(self.status(), scheduler_requests=2))
        record = dict(statuses=[])
        with self.assertRaises(ValueError):
            box.entry_status(client, T.by_label()[CMP], record)


if __name__ == '__main__':
    unittest.main()
