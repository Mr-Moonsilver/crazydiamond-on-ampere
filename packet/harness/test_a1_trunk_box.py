#!/usr/bin/env python3
"""q18a-a1-trunk box: the table, the reads, the smoke and the grade both ways, over synthetic arm records. CPU only; no pass here is box evidence. The arm itself: test_a1_trunk_router.ArmEnd, through the real router."""
import json
from pathlib import Path
import tempfile
import types
import unittest

import test_inwait_box as tbox   # first: puts tools/ and the engine sources on the path
import a1_trunk_arms as T
import e3_a1_trunk_box as box

tam, tap = tbox.tam, tbox.tap
START = 1_000_000_000_000


def yield_row(seq, start, *, arrival='stage0-ahead', served=True, reason='reopened'):
    times = dict(selected=start, drained=start + 300_000_000, row_closed=start + 310_000_000, admitted=start + 320_000_000, handback=start + 900_000_000, full=start + 950_000_000,
                 reenter=start + 1_100_000_000, resume=start + 1_200_000_000, bulk_reopened=start + 1_300_000_000)
    if not served:
        times = dict(selected=start, drained=start + 300_000_000)
    return dict(seq=seq, reason=reason, times=times, arrival=arrival, open_chunks=2, route_generation_selected=10 + seq, route_generation_end=12 + seq if served else 10 + seq)


def record_of(row, served=None):
    served = (2 if row['yield_on'] and row['plan'] else 0) if served is None else served
    returners = [dict(index=i, suffix_tokens=t, at_s=at, ttft_s=(1.5 if row['yield_on'] else 14.0), since_bulk_send_s=at, cached_tokens=153472, new_tokens=t + 65, prompt_tokens=153537 + t, output_completion=64, ok=True) for i, t, at in row['plan']]
    always = [i for i in range(8) if i not in {p[0] for p in row['plan']}]
    y0 = dict(seq=5, counters=dict(selected=5), end_reasons=dict(reopened=5), arrivals={'stage0-ahead': 5}, holds={}, current_seq=None)
    history = [yield_row(5 + k, START + k * 5_000_000_000) for k in range(1, served + 1)]
    y1 = dict(seq=5 + served, counters=dict(selected=5 + served), end_reasons=dict(reopened=5 + served), arrivals={'stage0-ahead': 5 + served}, holds={}, current_seq=None, history=history)
    return dict(schema=box.ARM_SCHEMA, kind='auto', label=row['label'], cohort=row['cohort'], c1=row['c1'], complete=True, reasons=[], wall_start_s=1000.0, wall_end_s=1060.0, yield_on=row['yield_on'], generation_requests_expected=row['generations'],
                yield_control=dict(enabled=row['yield_on'], confirmed=True, clock_proof=dict(inside=True)), inwait=dict(acknowledged=row['inwait'], early_acknowledged=row['next_levers']), attn_meta=tam.block(True, [40] * 4),
                outputs=dict(fresh=dict(tokens=1), append=dict(tokens=1), standing={'0': dict(tokens=1)}), progress_delta=dict(chunks=66, line_errors=0), returners=returners, always_on=always, append=dict(ttft_s=0.4),
                decode_only=dict(aggregate_tokens_per_s=450.0), stage_delta={'step_iw_applied': 30},
                cow_fast=dict(requested=True, counters=dict(calls={str(k): 2 for k in range(4)})), cow_host_filter=dict(requested=True, counters=dict(kept={str(k): 0 for k in range(4)}, skipped={str(k): 74 for k in range(4)}, unfiltered={str(k): 0 for k in range(4)})),
                client_basis=dict(prefill_tokens_per_s=3300.0, bulk_ttft_s=20.0 + (2.0 if row['yield_on'] and row['plan'] else 0.0), decode_window=dict(aggregate_tokens_per_s=57.0 * len(always), streams=[{}] * len(always))),
                statuses=[dict(label='before-fresh', route_generation=10, append_yield=y0), dict(label='after-fresh', route_generation=12 + 2 * served, append_yield=y1)], yield_stalls={str(5 + k): dict(full_return_stall_s=0.1, reentry_stall_s=0.1) for k in range(1, served + 1)})


def run_records():
    return tbox.decode_records() + [record_of(r) for r in T.arms()]


def grade_of(arms, watches=None, startup=tam.STARTUP, ingress=None, lines=(), span=None):
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / 'engine.log'
        log.write_text(''.join('INFO x %s\n' % line for line in startup) + ''.join(lines))
        if span:
            for a in arms:
                a.update(wall_start_s=span[0], wall_end_s=span[1])
        known = [a for a in arms if a.get('label')]
        return box.grade(arms, ingress if ingress is not None else {a['label']: tap.PASS for a in known}, tap.CHECKS, [str(log)], watches if watches is not None else {a['label']: tap.ALIVE for a in known})


class Table(unittest.TestCase):
    def test_the_order_and_the_matched_pairs(self):
        self.assertEqual(T.labels(), ['sm.g2-live-0', 'on.g2-live-0', 'off.g2-live-0', 'off.g2-live-1', 'on.g2-live-1', 'mn.g2-live-0', 'mo.g2-live-0', 'on.g2-solo-0'])
        self.assertEqual(T.labels('tail'), [])
        rows = T.by_label()
        self.assertEqual(T.matched_off(rows['on.g2-live-1']), 'off.g2-live-1')
        self.assertEqual(T.matched_off(rows['mn.g2-live-0']), 'mo.g2-live-0')
        self.assertIsNone(T.matched_off(rows['on.g2-solo-0']))
        self.assertIsNone(T.matched_off(rows['sm.g2-live-0']))

    def test_every_arm_is_inwaits_hf_and_the_axis_is_the_yield(self):
        # compose-4: the inwait table it merged has no ps1 row; hf is its base, cf with the host filter
        import inwait_arms as parent
        hf = next(r for r in parent.arms() if r['c1'] == 'hf')
        for r in T.arms():
            for name in ('attn', 'lazy_recv', 'decode_merge', 'inwait', 'next_levers', 'prepare_slices', 'score_warm', 'reserved_cap', 'overlap', 'defer_bind', 'stage_local', 'size'):
                self.assertEqual(r[name], hf[name], (r['label'], name))
            self.assertIs(r['cow_fast'], True, r['label'])
            self.assertIs(r['cow_host_filter'], True, r['label'])
            self.assertIs(r['yield_on'], T.YIELD[r['c1']])

    def test_inwaits_own_rows_name_the_host_filter(self):
        # compose-4: 5d's table carries the field on every row (it selects cn's off)
        import inwait_arms as parent
        self.assertTrue(all('cow_host_filter' in r for r in parent.arms()))

    def test_a_returner_is_a_sampled_stream_and_the_counts_add_up(self):
        rows = T.by_label()
        self.assertEqual(rows['on.g2-live-0']['generations'], 12)
        self.assertEqual(rows['mn.g2-live-0']['generations'], 13)
        self.assertEqual(rows['on.g2-solo-0']['generations'], 10)
        self.assertEqual(rows['on.g2-solo-0']['plan'], ())
        for r in T.arms():
            self.assertFalse({p[0] for p in r['plan']} & set(T.STANDING_GREEDY), r['label'])
            self.assertEqual(r['plan'], T.by_label()[T.label(T.MATCHED_OFF.get(r['c1'], r['c1']), 2, r['cohort'], r['index'])]['plan'] if T.matched_off(r) else r['plan'], r['label'])


class Returners(unittest.TestCase):
    def test_a_suffix_is_its_arms_and_streams_so_no_earlier_arm_cached_it(self):
        one = box.returner_text('history', 'on.g2-live-0', 7, 500)
        self.assertEqual(one, box.returner_text('history', 'on.g2-live-0', 7, 500), 'deterministic')
        self.assertTrue(one.startswith('history\n'), 'the agent\'s own prompt, then the suffix')
        others = [box.returner_text('history', 'off.g2-live-0', 7, 500), box.returner_text('history', 'on.g2-live-1', 7, 500), box.returner_text('history', 'on.g2-live-0', 6, 500)]
        for other in others:
            # Not even a shared first prefix unit (176 tokens).
            self.assertNotEqual(one.split()[1:9], other.split()[1:9])


class Rule1d(unittest.TestCase):
    def lines(self, edit=None):
        arms_read = {}
        for r in T.arms():
            rec = record_of(r)
            for ret in rec['returners']:
                ret['cached_tokens'] = None if r['yield_on'] else ret['cached_tokens']
            if edit:
                edit(r, rec)
            arms_read[r['label']] = box.arm_read(rec, 450.0)
        return box.lever_read(arms_read)

    def of(self, lines, line, arm='on.g2-live-0'):
        return [row for row in lines if row['line'] == line and row['arm'] == arm]

    def test_the_rule_is_read_beside_the_operators_budget(self):
        lines = self.lines()
        self.assertEqual({row['rule_s'] for row in self.of(lines, 'rule-1d-first-output')}, {2.5, 3.0})
        self.assertTrue(all(row['within'] for row in self.of(lines, 'rule-1d-first-output')))
        self.assertEqual(self.of(lines, 'rule-1d-decode')[0]['min_ratio'], 0.90)
        self.assertTrue(self.of(lines, 'first-output'))
        self.assertEqual({row['budget_s'] for row in self.of(lines, 'first-output')}, {2.0, 3.0})

    def test_a_returner_stock_prefilled_fails_its_line(self):
        self.assertTrue(self.of(self.lines(), 'rule-1d-no-stock-prefill')[0]['within'])

        def edit(row, rec):
            if row['label'] == 'on.g2-live-0':
                rec['returners'][0]['cached_tokens'] = 151000
        row = self.of(self.lines(edit), 'rule-1d-no-stock-prefill')[0]
        self.assertIs(row['within'], False)
        self.assertEqual(row['stock_usage'], [T.by_label()['on.g2-live-0']['plan'][0][0]])

    def test_a_returner_no_yield_served_fails_its_line(self):
        def edit(row, rec):
            if row['label'] == 'on.g2-live-0':
                rec['statuses'][1]['append_yield']['history'] = rec['statuses'][1]['append_yield']['history'][:1]
        self.assertIs(self.of(self.lines(edit), 'rule-1d-no-stock-prefill')[0]['within'], False)


class Tails(unittest.TestCase):
    A = types.SimpleNamespace(url='http://x', model='m')

    def post(self, base):
        """/tokenize of the fixture: the words of the suffix count one token each, on top of a long prompt."""
        self.calls = []

        def reply(url, path, body, timeout):
            self.calls.append(path)
            return dict(tokens=[0] * (base + len(body['prompt'].split())))
        return reply

    def test_the_box_tails_are_short_and_shaped_ones_fit(self):
        self.assertEqual((T.tail_rows(151905), T.tail_rows(152651), T.tail_rows(65445)), (16, 58, 148))
        for base in range(151000, 151000 + 2 * T.PREFIX_UNIT, 7):
            text, shape = box.shaped_returner_text(self.A, 'history', 'on.g2-live-0', 7, 500, False, post=self.post(base))
            self.assertGreaterEqual(shape['tail_rows'], T.MIN_TAIL, base)
            self.assertLessEqual(shape['tail_attempts'], T.TAIL_ATTEMPTS)
            self.assertEqual(self.calls, ['/tokenize'] * shape['tail_attempts'])
            self.assertTrue(text.startswith('history\n'))

    def test_a_short_one_stays_short(self):
        for base in range(151000, 151000 + 2 * T.PREFIX_UNIT, 7):
            _, shape = box.shaped_returner_text(self.A, 'history', 'mn.g2-live-0', 5, 1500, True, post=self.post(base))
            self.assertTrue(0 < shape['tail_rows'] < T.MIN_TAIL, base)

    def test_only_the_several_plan_keeps_one_short(self):
        rows = T.by_label()
        self.assertTrue(T.keeps_short_tail(rows['mn.g2-live-0'], 5))
        self.assertTrue(T.keeps_short_tail(rows['mo.g2-live-0'], 5))
        for label in ('sm.g2-live-0', 'on.g2-live-0', 'off.g2-live-0'):
            for row in rows[label]['plan']:
                self.assertFalse(T.keeps_short_tail(rows[label], row[0]))

    def test_a_bad_reply_stops_the_arm(self):
        with self.assertRaises(ValueError):
            box.shaped_returner_text(self.A, 'h', 'on.g2-live-0', 7, 500, False, post=lambda *_: {})


class Lookups(unittest.TestCase):
    def test_the_arm_delta_keeps_the_scan_apart(self):
        read = box.lookup_read(dict(calls=2, ns=10, scan_calls=1, scan_ns=5, max_ns=7), dict(calls=7, ns=1000010, scan_calls=2, scan_ns=5, max_ns=9000000))
        self.assertEqual((read['calls'], read['scan_calls'], read['other_calls'], read['mean_ms'], read['max_ms']), (5, 1, 4, 0.2, 9.0))
        self.assertIsNone(box.lookup_read(None, None), 'no timer yet: no bounds, or nothing offered')


class Reads(unittest.TestCase):
    def test_the_gap_counts_whole_across_the_window(self):
        ms = 10 ** 6
        state = types.SimpleNamespace(arrivals=[100 * ms, 200 * ms, 900 * ms, 1000 * ms])
        self.assertEqual(box.longest_gap_s(state, 300 * ms, 800 * ms), 0.7)
        self.assertIsNone(box.longest_gap_s(state, 50 * ms, 800 * ms), 'no arrival before the window')
        self.assertIsNone(box.longest_gap_s(state, 300 * ms, 1100 * ms), 'no arrival after the window')

    def test_the_stalls_need_the_clock_proof(self):
        state = types.SimpleNamespace(arrivals=[0, 10 * 10 ** 9])
        history = [yield_row(6, 10 ** 9)]
        self.assertEqual(box.yield_stalls(history, [state], False), {})
        self.assertEqual(box.yield_stalls(history, [state], True), {'6': dict(full_return_stall_s=10.0, reentry_stall_s=10.0)})

    def test_a_stream_without_an_arrival_makes_the_stall_unknown_not_zero(self):
        history = [yield_row(6, 10 ** 9)]
        self.assertEqual(box.yield_stalls(history, [types.SimpleNamespace(arrivals=[])], True), {'6': dict(full_return_stall_s=None, reentry_stall_s=None)})

    def test_the_yield_read_takes_only_this_arm_and_its_phases(self):
        row = T.by_label()['on.g2-live-0']
        rec = record_of(row)
        rec['statuses'][1]['append_yield']['history'].insert(0, yield_row(5, 1))       # an older arm's
        y = box.yield_read(rec)
        self.assertEqual([e['seq'] for e in y['yields']], [6, 7])
        self.assertEqual((y['served'], y['journal'], y['counters'], y['route_generation_window']), (2, 2, dict(selected=2), 6))
        first = y['yields'][0]
        self.assertEqual(first['steps']['selected>drained'], 0.3)
        self.assertEqual(first['selected_to_full_s'], 0.95)
        self.assertEqual(first['selected_to_reopened_s'], 1.3)
        self.assertEqual((first['arrival'], first['open_chunks'], first['route_generation_delta']), ('stage0-ahead', 2, 2))
        self.assertEqual(first['phases_seen'], list(box.PHASES))

    def test_a_yield_that_ended_in_the_drain_is_not_served(self):
        rec = record_of(T.by_label()['on.g2-live-0'], served=0)
        rec['statuses'][1]['append_yield'].update(history=[yield_row(6, START, served=False, reason='drain-no-append')], seq=6, counters=dict(selected=6))
        y = box.yield_read(rec)
        self.assertEqual((y['served'], y['journal'], y['yields'][0]['reason']), (0, 1, 'drain-no-append'))
        self.assertEqual(y['yields'][0]['route_generation_delta'], 0)

    def test_the_budget_lines(self):
        reads = {r['label']: box.arm_read(record_of(r), 450.0) for r in T.arms()}
        lines = box.lever_read(reads)
        first = [x for x in lines if x['line'] == 'first-output']
        self.assertEqual(sorted({x['arm'] for x in first}), ['mn.g2-live-0', 'on.g2-live-0', 'on.g2-live-1', 'sm.g2-live-0'])
        self.assertTrue(all(x['within'] for x in first))
        self.assertEqual({(x['suffix_tokens'], x['budget_s']) for x in first}, {(500, 2.0), (1500, 3.0)})
        bulk = {x['arm']: x for x in lines if x['line'] == 'bulk-complete'}
        self.assertEqual(sorted(bulk), ['mn.g2-live-0', 'on.g2-live-0', 'on.g2-live-1'])
        self.assertIs(bulk['on.g2-live-0']['within'], True)
        decode = {x['arm']: x for x in lines if x['line'] == 'decode'}
        self.assertEqual(decode['on.g2-live-0']['ratio'], 1.0)

    def test_a_budget_miss_is_a_read_not_a_verdict(self):
        arms = run_records()
        rec = next(a for a in arms if a['label'] == 'on.g2-live-0')
        rec['returners'][0]['ttft_s'] = 9.0
        rec['client_basis']['bulk_ttft_s'] = 40.0
        rec['client_basis']['decode_window']['aggregate_tokens_per_s'] = 10.0
        result = grade_of(arms)
        self.assertEqual(result['verdict'], 'PASS', (result['failures'], result['missing']))
        misses = [x for x in result['lever_read'] if x.get('arm') == 'on.g2-live-0' and x['within'] is False]
        self.assertEqual(sorted(x['line'] for x in misses), ['bulk-complete', 'decode', 'first-output', 'rule-1d-decode', 'rule-1d-first-output', 'rule-1d-no-stock-prefill'])
        self.assertIs(result['is_budget_gate'], False)


class Grade(unittest.TestCase):
    def test_the_run_passes_with_every_section_and_a_report(self):
        result = grade_of(run_records())
        self.assertEqual(result['verdict'], 'PASS', (result['failures'], result['missing']))
        self.assertEqual(sorted(result['arms']), sorted(T.labels()))
        self.assertEqual(result['cells']['on live']['served'], [2, 2])
        text = box.report(result)
        self.assertTrue(text.startswith('# q18a-a1-trunk report'))
        for mark in ('## Budget lines', '## Yields', '## Returners', 'selected>drained', 'on.g2-solo-0'):
            self.assertIn(mark, text)

    def test_an_off_arm_that_selected_a_yield_fails(self):
        arms = run_records()
        rec = next(a for a in arms if a['label'] == 'off.g2-live-0')
        rec['statuses'][1]['append_yield'].update(counters=dict(selected=6), history=[yield_row(6, START)])
        rec['statuses'][1]['route_generation'] = 14
        result = grade_of(arms)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('off.g2-live-0:yield-off-selected-a-yield', result['failures'])

    def test_an_on_arm_that_served_none_is_inc(self):
        arms = [a if a.get('label') != 'on.g2-live-1' else record_of(T.by_label()['on.g2-live-1'], served=0) for a in run_records()]
        result = grade_of(arms)
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('on.g2-live-1:yield-on-served-none', result['missing'])

    def test_route_generations_that_do_not_add_up_fail(self):
        arms = run_records()
        next(a for a in arms if a['label'] == 'on.g2-live-1')['statuses'][1]['route_generation'] = 13
        result = grade_of(arms)
        self.assertIn('on.g2-live-1:route-generations-3-for-2-served', result['failures'])

    def test_a_yield_open_at_the_end_fails(self):
        arms = run_records()
        next(a for a in arms if a['label'] == 'on.g2-live-1')['statuses'][1]['append_yield']['current_seq'] = 9
        self.assertIn('on.g2-live-1:yield-open-at-the-arm-end', grade_of(arms)['failures'])

    def test_a_death_watch_and_a_signature_fail(self):
        arms = run_records()
        watches = {a['label']: tap.ALIVE for a in arms}
        watches['on.g2-live-1'] = dict(verdict='DEAD')
        self.assertIn('engine-death-watch:on.g2-live-1', grade_of(arms, watches)['failures'])
        arms = run_records()
        result = grade_of(arms, startup=list(tam.STARTUP), lines=['2026-11-04T10:00:30.000000000Z Traceback (most recent call last):\n'], span=(1793786400.0 + 10.0, 1793786400.0 + 60.0))
        self.assertTrue([f for f in result['failures'] if f.startswith('engine-signature:') and 'Traceback' in f], result['failures'])

    def test_a_missing_arm_an_incomplete_arm_and_a_bad_ingress_are_inc(self):
        arms = [a for a in run_records() if a.get('label') != 'mo.g2-live-0']
        next(a for a in arms if a['label'] == 'off.g2-live-1').update(complete=False, reasons=['returner-not-complete:7'])
        ingress = {a['label']: tap.PASS for a in arms}
        ingress['on.g2-live-1'] = dict(verdict='INVALID')
        result = grade_of(arms, ingress=ingress)
        self.assertEqual(result['verdict'], 'INC')
        for mark in ('mo.g2-live-0:no-arm-record', 'off.g2-live-1:incomplete:returner-not-complete:7', 'on.g2-live-1:ingress-not-the-arm'):
            self.assertIn(mark, result['missing'])

    def test_a_fallback_is_inc(self):
        arms = run_records()
        next(a for a in arms if a['label'] == 'on.g2-live-0')['statuses'][1]['append_yield']['counters']['handback_expired'] = 1
        self.assertIn('on.g2-live-0:a-yield-fell-back', grade_of(arms)['missing'])


class Smoke(unittest.TestCase):
    def rec(self):
        return record_of(T.by_label()[T.SMOKE_ARM])

    def test_an_engaged_arm_passes(self):
        self.assertEqual(box.smoke_check(self.rec()), (True, []))

    def test_each_failure_has_its_name(self):
        for edit, want in ((lambda r: r.update(complete=False), 'smoke-arm-not-complete'),
                           (lambda r: r['outputs'].pop('append'), 'smoke-request-shape-not-observed'),
                           (lambda r: r.update(generation_requests_expected=10), 'smoke-request-shape-differs'),
                           (lambda r: r['yield_control'].update(confirmed=False), 'smoke-yield-control-not-acknowledged'),
                           (lambda r: r['inwait'].update(acknowledged=False), 'smoke-control-not-acknowledged'),
                           (lambda r: r['attn_meta'].update(select=None), 'smoke-select-not-sent'),
                           (lambda r: r['progress_delta'].update(chunks=0), 'smoke-no-native-chunks'),
                           (lambda r: r.pop('cow_host_filter'), 'smoke-host-filter-not-engaged'),
                           (lambda r: r['cow_host_filter']['counters']['skipped'].update({'2': 0}), 'smoke-host-filter-not-engaged'),
                           (lambda r: r['cow_fast']['counters']['calls'].update({'0': 0}), 'smoke-cow-fast-not-engaged'),
                           (lambda r: r['statuses'][1]['append_yield'].update(history=[]), 'smoke-no-yield-served'),
                           (lambda r: r['statuses'][1]['append_yield']['counters'].update(reentry_refused=1), 'smoke-a-yield-fell-back'),
                           (lambda r: r['returners'][0].update(ok=False), 'smoke-a-returner-did-not-finish')):
            rec = self.rec()
            edit(rec)
            ok, reasons = box.smoke_check(rec)
            self.assertFalse(ok, want)
            self.assertIn(want, reasons)

    def test_an_unreadable_record_does_not_pass(self):
        self.assertEqual(box.smoke_check(None), (False, ['smoke-record-unreadable']))

    def test_the_cli_writes_its_record_and_says_so_in_the_rc(self):
        with tempfile.TemporaryDirectory() as tmp:
            arm, out = Path(tmp) / 'a.json', Path(tmp) / 's.json'
            arm.write_text(json.dumps(self.rec()))
            self.assertEqual(box.main(['smoke', '--arm', str(arm), '--out', str(out)]), 0)
            self.assertIs(json.loads(out.read_text())['engaged'], True)
            bad, out2 = self.rec(), Path(tmp) / 's2.json'
            bad['statuses'][1]['append_yield']['history'] = []
            arm.write_text(json.dumps(bad))
            self.assertEqual(box.main(['smoke', '--arm', str(arm), '--out', str(out2)]), 2)


if __name__ == '__main__':
    unittest.main()
