#!/usr/bin/env python3
"""decode-merge probe: the exploratory reader (tools/q2024p_j2_dmp_reader.py) on synthetic records.

Pure CPU; records are written to fresh mkdtemp directories (left in place, never removed). Line shapes are copied from
a real j2 engine-stream.log (native-gap, memory point, handler 'm' rows, worker anatomy 'c' rows, flash-sched stats);
the decode-liveness and submit-cpu lines are this branch's (stage_overlap.LIVENESS_LINE, executor_hooks submit-cpu).
No engine, torch or box access.
"""
import datetime
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout

HERE = Path(__file__).resolve().parent
for extra in (HERE.parent / 'tools', HERE / 'tools'):   # source tree; the bundle keeps tools/ beside the suites
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))

import q2024p_j2_dmp_reader as reader  # noqa: E402

T0 = 1794436000.0            # replay start (wall) of the native arm
OFFSET = 1794000000.0        # wall - monotonic
SP0 = T0 - 1000.0            # replay start of the sp arm


def iso(wall):
    whole = int(wall)
    frac = int(round((wall - whole) * 1e9))
    return datetime.datetime.fromtimestamp(whole, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + '.%09dZ' % frac


def mono_ns(wall):
    return int(round((wall - OFFSET) * 1e9))


def gap(wall, event, **fields):
    d = dict(pid=85, seq=1, sha='x', event=event, observed_ns=mono_ns(wall), schema='s5-native-gap/1', **fields)
    return '%s flash-s5 native-gap %s\n' % (iso(wall + 0.0005), json.dumps(d))


def memory(wall, point, rank, txn, c0, c1, ooms=0):
    return ('%s flash-s5 memory point=%s rank=%d txn_seq=%d chunk=%d-%d allocated=1 reserved=1 inactive_split=0 '
            'alloc_retries=0 ooms=%d free=1 total=2 pid=95 seq=1 sha=x\n' % (iso(wall), point, rank, txn, c0, c1, ooms))


def handler(wall, rank, forward_ms):
    base = mono_ns(wall)
    row = [1, 'e', 0, 4, 0, 776, 776761, 648419, 26180, int(forward_ms * 1e6), None, None, None, None, []]
    d = dict(pid=95, seq=5, sha='x', base_ns=base, line=1, part=0, parts=1, rank=rank, rows=dict(m=[row]), schema='h')
    return '%s flash-s5 handler %s\n' % (iso(wall + 0.01), json.dumps(d))


X_FIELDS = ['op_seq', 'kind', 'layer', 'refused', 't_host', 't_start', 't_end']     # a real j2 worker anatomy line's


def anatomy(wall, rank, c0, c1, busy_ms, window_ms, waits_ms=(), idle_ms=0.0):
    """A worker chunk row; x rows with device start - host enqueue = each of waits_ms (us offsets, the real layout)."""
    base = mono_ns(wall)
    c = [33, 33, 0, 0, 0, int(window_ms * 1000), int(busy_ms * 1000), int(idle_ms * 1000), 0, 0, 0, None, 0]
    x = [[i, 'layer', i, 0, 1000 * i, 1000 * i + int(w * 1000), 1000 * i + int(w * 1000) + 50] for i, w in enumerate(waits_ms)]
    d = dict(pid=101, seq=8, sha='x', base_ns=base, chunk_start=c0, chunk_end=c1, rank=rank,
             fields=dict(x=X_FIELDS), rows=dict(c=[c], x=x))
    return '%s flash-s5 anatomy %s\n' % (iso(wall + 0.5), json.dumps(d))


def sched(wall, step, dec_even, dec_odd, cap_skips, even, odd):
    stats = {'D': 0, 'P': 0, 'plain': step, 'dec_even': dec_even, 'dec_odd': dec_odd, 'cap_skips': cap_skips,
             'parity_changes': 0}
    return ('%s (EngineCore pid=85) INFO 11-11 22:20:53 [scheduler.py:1500] flash-sched stats step %d: %r cohorts '
            'even=%d odd=%d none=0 decode_share_3s=None\n' % (iso(wall), step, stats, even, odd))


def submit(wall, rank, n, wall_ns, cpu_ns):
    d = dict(n=n, wall_ns=wall_ns, cpu_ns=cpu_ns, max_wall_ns=9_000_000, max_cpu_ns=1, by_wall_ms={}, rank=rank)
    return '%s flash-s5 submit-cpu %s\n' % (iso(wall), json.dumps(d, sort_keys=True))


def turn(agent, t, ttft, period_ms, count):
    """A journal turn: arrivals_ms are ms since the replay start (first = t + ttft), as compose4_replay writes them."""
    first = int(round((t + ttft) * 1000))
    arrivals = [first + i * period_ms for i in range(count)]
    return dict(agent=agent, session=1, kind='fresh', t=t, ttft=ttft, seconds=arrivals[-1] / 1000.0 - t + 0.001,
                completion_tokens=count, arrivals_ms=arrivals, prompt_tokens=81000, cached_tokens=1000)


def arm(label, start, *, native_off, merge, turns_done, liveness=None, guard=None, steps=0):
    statuses = [dict(label='entry'), dict(label='before'), dict(label='after')]
    if liveness is not None:
        statuses[1]['decode_liveness'], statuses[2]['decode_liveness'] = liveness
    rec = dict(label=label, kind='replay', complete=True, reasons=[],
               native_off=dict(requested=native_off),
               replay_config=dict(AGENTS=8, DURATION=480.0, WARMUP=0.0, SEED=1), replay_start_s=start,
               replay_end_s=start + 480.0,
               replay_summary=dict(warm_turns=turns_done + 2, turns_finished_after_window=2),
               step=dict(requested_merge=merge, acknowledged_merge=merge),
               stage_delta=dict(step_merge_steps=steps, step_merge_on=1, step_merge_off=1, step_lazy_marked=0),
               statuses=statuses)
    if guard is not None:
        rec['decode_liveness_guard'] = guard
    return rec


def write_record(root, *, oom=0):
    lines = []
    # sp arm window [SP0, SP0+480): stock; a sched line before and inside
    lines.append(sched(SP0 - 5, 200, 10, 10, 0, 4, 4))
    lines.append(sched(SP0 + 100, 1200, 1010, 990, 5, 4, 4))
    # native arm: FULL until armed at T0+100; a bulk txn T0+200 .. T0+260 (rank 0 opens; ranks release); disarmed
    lines.append(gap(T0 - 10, 'control_submit_begin'))
    lines.append(gap(T0 + 100, 'admission_armed'))
    lines.append(memory(T0 + 200, 'open', 0, 7, 0, 2048, ooms=oom))
    lines.append(memory(T0 + 200, 'open', 2, 7, 0, 2048))
    lines.append(memory(T0 + 260, 'released', 0, 7, 0, 2048))
    lines.append(memory(T0 + 260.5, 'released', 2, 7, 0, 2048))
    lines.append(memory(T0 + 300, 'open', 0, 8, 2048, 2148))         # a tiny txn T0+300 .. T0+310
    lines.append(memory(T0 + 310, 'released', 0, 8, 2048, 2148))
    lines.append(gap(T0 + 400, 'admission_disarmed', reason='route-returned-full'))
    for k in range(10):
        lines.append(handler(T0 + 210 + k, 0, 14.0 + k * 0.1))
        lines.append(handler(T0 + 210 + k, 2, 13.0))
    lines.append(handler(T0 + 2000, 0, 99.0))                        # outside every window
    lines.append(anatomy(T0 + 201, 0, 0, 2048, 230.0, 243.0, waits_ms=(2.0, 4.0, 6.0), idle_ms=12.0))
    lines.append(anatomy(T0 + 201, 2, 0, 2048, 245.0, 300.0))
    lines.append(anatomy(T0 + 301, 0, 2048, 2148, 20.0, 25.0))      # not bulk
    lines.append(sched(T0 + 50, 3000, 2000, 1000, 10, 8, 0))
    lines.append(sched(T0 + 450, 5000, 4000, 1100, 30, 7, 1))
    lines.append(submit(T0 - 1, 0, 100, 1_000_000_000, 400_000_000))
    lines.append(submit(T0 + 260, 0, 150, 1_500_000_000, 600_000_000))
    lines.append('%s flash-s5 decode-liveness {"rid":"r1","stall_ms":10500}\n' % iso(T0 + 270))
    (root / 'engine-stream.log').write_text(''.join(lines))
    sp_turns = [turn(0, 10.0, 1.0, 15, 200), turn(1, 10.0, 1.0, 15, 200)]
    n3_turns = [turn(0, 0.5, 0.5, 15, 3000),                  # full (0.5 .. 100): 15 ms gaps at 2 live
                turn(1, 0.5, 0.5, 15, 3000),
                turn(2, 199.0, 2.0, 20, 2500),                # bulk window 200..260: 20 ms gaps, 2 live
                turn(3, 199.0, 2.0, 20, 2500)]
    for label, start, rows, native_off, merge, done, live, guard in (
            ('sp.g2-live-0', SP0, sp_turns, True, 'off', 100, None, None),
            ('n3.g2-live-0', T0, n3_turns, False, 'chunk', 120,
             (dict(trips=0, live_max_stall_ms=1), dict(trips=1, live_max_stall_ms=2)),
             dict(trip=dict(classification='ENGINE', detail='engine-trip', wall_s=None), harness=None, trips_delta=1,
                  max_live_stall_ms=2, polls=30, poll_failures=0, ended_replay=False))):
        (root / ('arm-%s.json' % label)).write_text(json.dumps(arm(label, start, native_off=native_off, merge=merge,
                                                                  turns_done=done, liveness=live, guard=guard)))
        with (root / ('arm-%s.json.turns.jsonl' % label)).open('w') as handle:
            handle.write(json.dumps(dict(label=label, header=True, replay_start_s=start)) + '\n')
            for row in rows:
                handle.write(json.dumps(row) + '\n')
    (root / 'arm-pf.g2-live-0.json').write_text(json.dumps(dict(label='pf.g2-live-0', kind='prefill')))
    (root / 'compose5-grade.json').write_text(json.dumps(dict(
        verdict='PASS', numerics_o3=dict(verdict='WITHIN'),
        coverage={'n3.g2-live-0': dict(live_native_share_strict=0.99, all_native_share_strict=0.98)})))
    (root / 'queue-grade.json').write_text(json.dumps(dict(verdict='PASS', rc=0, engine_crashed=False)))
    return root


class Reader(unittest.TestCase):
    def setUp(self):
        self.root = write_record(Path(tempfile.mkdtemp(prefix='dmp-reader-')))
        self.res = reader.collect(self.root, 'syn')

    def test_offset_and_arms(self):
        self.assertAlmostEqual(self.res['engine']['offset_s'], OFFSET, delta=0.01)
        self.assertEqual(list(self.res['arms']), ['sp.g2-live-0', 'n3.g2-live-0'])   # replay order, pf skipped
        self.assertEqual(self.res['notes'], [])

    def test_gaps_by_state_and_live(self):
        gaps = self.res['arms']['n3.g2-live-0']['gaps']
        self.assertEqual(gaps['full|2']['p50_ms'], 15)
        self.assertEqual(gaps['txn_bulk|2']['p50_ms'], 20)
        sp = self.res['arms']['sp.g2-live-0']
        self.assertEqual(set(k.split('|')[0] for k in sp['gaps']), {'full'})           # native off: all full
        self.assertEqual(sp['gaps']['full|2']['p50_ms'], 15)
        states = self.res['arms']['n3.g2-live-0']['state_seconds']
        self.assertAlmostEqual(states['txn_bulk'], 60.5, delta=0.2)
        self.assertAlmostEqual(states['txn_tiny'], 10.0, delta=0.2)
        self.assertAlmostEqual(states['full'], 100.0 + 80.0, delta=0.2)

    def test_forward_busy_sched_submit_liveness(self):
        a = self.res['arms']['n3.g2-live-0']
        self.assertEqual(a['forward']['0']['n'], 10)
        self.assertEqual(a['forward']['0']['bulk_n'], 10)
        self.assertAlmostEqual(a['forward']['0']['mean_ms'], 14.45, places=3)
        self.assertEqual(a['forward']['2']['p50_ms'], 13.0)
        self.assertEqual(a['bulk_busy']['0'], dict(chunks=1, busy_mean_ms=230.0, window_mean_ms=243.0))
        self.assertEqual(a['bulk_busy']['2']['busy_mean_ms'], 245.0)
        s = a['sched']
        self.assertEqual((s['dec_even'], s['dec_odd'], s['cap_skips'], s['steps']), (2990, 110, 25, 3800))
        self.assertEqual((s['cohort_even_mean'], s['cohort_odd_mean']), (7.5, 0.5))
        self.assertEqual(a['submit_cpu']['0']['calls'], 50)
        self.assertEqual((a['submit_cpu']['0']['wall_ms'], a['submit_cpu']['0']['cpu_ms']), (500.0, 200.0))
        self.assertEqual(a['liveness_lines'], 1)
        self.assertEqual(a['liveness_status']['after']['trips'], 1)
        self.assertEqual(a['liveness_guard']['trip']['classification'], 'ENGINE')

    def test_agent_seconds_and_queue_wait(self):
        a = self.res['arms']['n3.g2-live-0']
        ags = a['agent_seconds']
        self.assertEqual(ags['turns'], 4)
        self.assertAlmostEqual(ags['ttft_s'], 5.0, places=3)                    # 0.5 + 0.5 + 2 + 2
        self.assertEqual(ags['output_tokens'], 2 * 3000 + 2 * 2500)
        self.assertEqual(ags['uncached_prompt_tokens'], 4 * 80000)
        self.assertAlmostEqual(ags['ttft_ms_per_1k_uncached'], 5.0 * 1e6 / 320000, delta=0.01)
        q = a['queue_wait']['0']
        self.assertEqual((q['chunks'], q['sampled_chunks'], q['commands'], q['wait_per_chunk_ms'], q['wait_per_command_ms'],
                          q['idle_per_chunk_ms']), (2, 1, 3, 12.0, 4.0, 6.0))   # the tiny chunk: no x rows (unsampled), idle 0
        text = reader.render([self.res])
        self.assertIn('native queue rank0:wait/chunk=12.0 wait/cmd=4.0', text)
        self.assertIn('agent-s ttft=5.0', text)
        self.assertLess(text.index('turns/h by arm:'), text.index('x sp by arm:'))

    def test_record_fields_and_render(self):
        self.assertEqual(self.res['arms']['n3.g2-live-0']['x_sp'], 1.2)
        self.assertIsNone(self.res['arms']['sp.g2-live-0']['native_share'])
        self.assertEqual(self.res['arms']['n3.g2-live-0']['native_share']['live_native_share_strict'], 0.99)
        self.assertEqual((self.res['numerics'], self.res['queue_rc'], self.res['engine']['oom_max']), ('WITHIN', 0, 0))
        text = reader.render([self.res])
        self.assertIn('gap txn_bulk', text)
        self.assertIn('(fit 25.8)', text)                 # 21.7 + 2.07 * 2
        self.assertIn('"classification": "ENGINE"', text)

    def test_oom_and_cli(self):
        root = write_record(Path(tempfile.mkdtemp(prefix='dmp-reader-oom-')), oom=2)
        out = root / 'read.json'
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(reader.main(['x=%s' % root, '--json-out', str(out)]), 0)
        self.assertIn('EXPLORATORY no rule, no thresholds, no verdict', buf.getvalue())
        doc = json.loads(out.read_text())
        self.assertEqual(doc['results'][0]['engine']['oom_max'], 2)
        with self.assertRaises(FileExistsError):              # never overwrites
            reader.main(['x=%s' % root, '--json-out', str(out)])


class EngineLogName(unittest.TestCase):
    """Brief 11: window 13's fetched records hold the engine stream as engine.log (the record's final capture) and no
    engine-stream.log, so every arm read 'arm-unread' and the reader noted 'missing-engine-stream.log'."""

    def fetched(self, keep_stream=False, empty_stream=False):
        root = write_record(Path(tempfile.mkdtemp(prefix='dmp-reader-fetch-')))
        stream = root / 'engine-stream.log'
        if keep_stream:
            (root / 'engine.log').write_text('2026-11-13T19:52:57.200440492Z an earlier capture\n')
        else:
            stream.rename(root / 'engine.log')
            if empty_stream:
                stream.write_text('')
        return root

    def test_a_fetched_record_reads_engine_log(self):
        res = reader.collect(self.fetched(), 'fetch')
        self.assertEqual((res['notes'], res['engine_log']), ([], 'engine.log'))
        self.assertAlmostEqual(res['engine']['offset_s'], OFFSET, delta=0.01)
        self.assertIn('gaps', res['arms']['n3.g2-live-0'])
        res = reader.collect(self.fetched(empty_stream=True), 'empty')     # a present stream is read as before
        self.assertEqual(res['engine_log'], 'engine-stream.log')
        self.assertIn('no-native-gap-lines: engine times not mapped', res['notes'])

    def test_the_box_stream_wins(self):
        res = reader.collect(self.fetched(keep_stream=True), 'box')
        self.assertEqual((res['notes'], res['engine_log']), ([], 'engine-stream.log'))

    def test_neither_is_still_noted(self):
        root = self.fetched()
        (root / 'engine.log').rename(root / 'engine.log.kept')
        res = reader.collect(root, 'none')
        self.assertIn('missing-engine-stream.log', res['notes'])
        self.assertIsNone(res['engine_log'])
        self.assertTrue(all(n.startswith('arm-unread:') for n in res['notes'] if n != 'missing-engine-stream.log'))

    def test_mutant_without_the_fallback_is_caught(self):
        with mock.patch.object(reader, 'ENGINE_LOGS', ('engine-stream.log',)):
            res = reader.collect(self.fetched(), 'fetch')
        self.assertIn('missing-engine-stream.log', res['notes'])


if __name__ == '__main__':
    unittest.main()
