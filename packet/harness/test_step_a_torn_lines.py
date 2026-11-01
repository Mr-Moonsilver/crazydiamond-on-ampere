#!/usr/bin/env python3
"""The torn-line rule, on real-format progress lines through the real assemble and progress_receipts."""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest

import e3_native41_progress_box as progress
import step_a_torn_lines as torn

CHUNK = 1024
ROWS = {k: list(v) for k, v in progress.ROWS.items()}


def worker_row(seq, at):
    return [seq, 'flash_s5_submit', at, at + 1, at + 2, 0, 1]


class Log:
    """A run's engine log: for each chunk, an engine flush and four worker flushes."""

    def __init__(self, chunks=8, span=100.0):
        self.chunks = chunks
        self.span = span
        self.lines = []
        self.records = {}
        seq = 1000
        for index in range(chunks):
            stamp = 1000.0 + index * span / chunks
            start = index * CHUNK
            seq += 1
            at = seq * 1000
            recv = [at + 2, at + 3, at + 4, at + 5]
            row = [seq, 'segment', 'layer', 1, 0, 0, 'tick', at, at + 1] + recv + [at + 6, 'accepted', 0]
            self.add(stamp, dict(side='engine', rank=None, line=index, rows=dict(n=[row]), chunk_start=start,
                                 chunk_end=start + CHUNK, decode_live=8, mode='fused', open_rows=0, txn=index))
            for rank in range(4):
                self.add(stamp, dict(side='worker', rank=rank, line=index, rows=dict(r=[worker_row(seq, at + 1 + rank)])))

    def add(self, stamp, body, parts=1, part=0):
        body = dict(body, base_ns=0, dropped=0, fields=ROWS, part=part, parts=parts, schema=progress.PROGRESS_SCHEMA)
        self.lines.append((stamp, body))

    def text(self, tear=(), drop=()):  # keys: (side, rank, line, part)
        out = []
        for stamp, body in self.lines:
            key = (body['side'], body['rank'], body['line'], body['part'])
            if key in drop:
                continue
            who = '(Worker_PP0_TP0_EP0 pid=9)' if body['side'] == 'worker' else '(EngineCore pid=1)'
            when = self.when(stamp)
            text = json.dumps(body, sort_keys=True, separators=(',', ':'))
            if key in tear:
                text = text[:-20]
            out.append('%s %s INFO [executor_hooks.py:1303] flash-s5 progress %s\n' % (when, who, text))
        return ''.join(out)

    @staticmethod
    def when(stamp):
        import datetime
        moment = datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc)
        return moment.strftime('%Y-%m-%dT%H:%M:%S.%f000Z')


def arm(chunks=8, span=100.0):
    return dict(label='fused-live-0', mode='fused', cohort='live', wall_start_s=999.0, wall_end_s=1000.0 + span,
                progress_delta=dict(chunks=chunks), fresh=dict(prompt_tokens_observed=chunks * CHUNK + 1))


class Rule(unittest.TestCase):
    def grade(self, text, arms=None):
        work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, work)
        path = os.path.join(work, 'engine-current.log')
        with open(path, 'w') as handle:
            handle.write(text)
        arms = arms or [arm()]
        return torn.receipts_for(arms, [path])[1]

    def test_a_clean_log_is_what_progress_receipts_gives(self):
        text = Log().text()
        work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, work)
        path = os.path.join(work, 'engine-current.log')
        with open(path, 'w') as handle:
            handle.write(text)
        records, errors = progress.read_lines([path])
        base = progress.progress_receipts([arm()], records, errors)
        graded = self.grade(text)
        self.assertEqual(errors, [])
        self.assertEqual(graded['problems'], base['problems'])
        one, other = graded['arms']['fused-live-0'], base['arms']['fused-live-0']
        self.assertEqual(one['torn'], dict(lost_flushes=0, torn_lines=0, chunks_dropped=0, limit=2, over_limit=False))
        self.assertEqual(one['reasons'], other['reasons'])
        for key in ('chunks', 'bands', 'all_counted', 'counted_chunks', 'missing_worker_rows'):
            self.assertEqual(one[key], other[key], key)

    def test_one_torn_worker_line_is_counted_and_only_its_chunk_drops(self):
        graded = self.grade(Log().text(tear={('worker', 2, 3, 0)}))
        row = graded['arms']['fused-live-0']
        self.assertEqual(graded['problems'], [])
        self.assertEqual(row['torn']['lost_flushes'], 1)
        self.assertEqual(row['torn']['torn_lines'], 1)
        self.assertNotIn('worker-rows-missing', row['reasons'])
        self.assertNotIn('torn-progress-flushes-over-limit', row['reasons'])
        self.assertEqual(graded['torn']['detail'][0]['rank'], 2)
        # The critical rank in this log is rank 3, so a rank 2 row is not a missing row.
        self.assertEqual(row['counted_chunks'], 8)

    def test_a_lost_flush_on_the_critical_rank_drops_that_chunk_from_the_breakdown(self):
        graded = self.grade(Log().text(tear={('worker', 3, 3, 0)}))
        row = graded['arms']['fused-live-0']
        self.assertEqual(row['reasons'], [])
        self.assertEqual(row['torn']['chunks_dropped'], 1)
        self.assertEqual(row['counted_chunks'], 7)
        self.assertEqual(row['all_counted']['chunks'], 7)
        self.assertNotIn(3 * CHUNK, [c['chunk_start'] for c in row['chunks']])

    def test_a_flush_missing_a_part_is_a_lost_flush(self):
        log = Log()
        log.add(1050.0, dict(side='worker', rank=3, line=99, rows=dict(r=[])), parts=2, part=1)
        graded = self.grade(log.text())
        row = graded['arms']['fused-live-0']
        self.assertEqual(graded['problems'], [])
        self.assertEqual(row['torn']['lost_flushes'], 1)
        self.assertEqual(row['reasons'], [])

    def test_at_the_limit_passes_and_above_it_is_inconclusive(self):
        at_limit = self.grade(Log().text(tear={('worker', 3, 2, 0), ('worker', 3, 5, 0)}))['arms']['fused-live-0']
        self.assertEqual(at_limit['torn']['lost_flushes'], 2)
        self.assertEqual(at_limit['reasons'], [])
        over = self.grade(Log().text(tear={('worker', 3, 1, 0), ('worker', 3, 3, 0), ('worker', 3, 5, 0)}))['arms']['fused-live-0']
        self.assertEqual(over['torn']['lost_flushes'], 3)
        self.assertTrue(over['torn']['over_limit'])
        self.assertEqual(over['reasons'], ['torn-progress-flushes-over-limit'])

    def test_a_missing_row_with_no_lost_flush_still_fails_the_arm(self):
        # The flush is gone with no torn line and no surviving part: nothing witnesses a tear.
        graded = self.grade(Log().text(drop={('worker', 3, 3, 0)}))
        row = graded['arms']['fused-live-0']
        self.assertIn('worker-rows-missing', row['reasons'])
        self.assertEqual(row['missing_worker_rows'], 1)
        self.assertEqual(row['torn']['lost_flushes'], 0)

    def test_a_missing_row_outside_the_lost_flush_still_fails_the_arm(self):
        graded = self.grade(Log().text(tear={('worker', 3, 2, 0)}, drop={('worker', 3, 6, 0)}))
        row = graded['arms']['fused-live-0']
        self.assertIn('worker-rows-missing', row['reasons'])
        self.assertEqual(row['missing_worker_rows'], 1)
        self.assertEqual(row['torn']['chunks_dropped'], 1)

    def test_a_torn_line_with_no_rank_left_explains_nothing(self):
        log = Log()
        text = log.text()
        lines = text.splitlines(True)
        victim = next(i for i, line in enumerate(lines) if '(Worker_' in line and '"line":4,' in line and '"rank":3' in line)
        lines[victim] = lines[victim].split('flash-s5 progress ')[0] + 'flash-s5 progress {"base_ns":0,"dro\n'
        graded = self.grade(''.join(lines))
        row = graded['arms']['fused-live-0']
        self.assertIn('worker-rows-missing', row['reasons'])
        self.assertEqual(graded['torn']['torn_lines'], 1)
        self.assertEqual(row['torn']['lost_flushes'], 1)

    def test_a_torn_engine_line_drops_its_chunk_and_the_interval_across_it(self):
        graded = self.grade(Log().text(tear={('engine', None, 3, 0)}))
        self.assertEqual(graded['torn']['detail'][0]['side'], 'engine')
        row = graded['arms']['fused-live-0']
        self.assertEqual(row['reasons'], [])
        self.assertEqual(row['torn']['chunks_dropped'], 1)
        self.assertEqual(row['counted_chunks'], 7)
        before = next(c for c in row['chunks'] if c['chunk_end'] == 3 * CHUNK)
        self.assertNotIn('open_interval_s', before)

    def test_two_engine_chunks_lost_without_a_torn_line_still_fail(self):
        log = Log()
        graded = self.grade(log.text(drop={('engine', None, 2, 0), ('engine', None, 3, 0)}))
        reasons = graded['arms']['fused-live-0']['reasons']
        self.assertIn('progress-lines-differ-from-the-chunk-counter', reasons)
        self.assertIn('native-chunks-not-row-0-through-n-minus-1', reasons)

    def test_other_progress_problems_stay(self):
        log = Log()
        text = log.text().replace('"schema":"%s"' % progress.PROGRESS_SCHEMA, '"schema":"other"', 1)
        self.assertIn('progress-line-schema', self.grade(text)['problems'])

    def test_a_lost_flush_outside_every_arm_is_reported_not_charged(self):
        log = Log()
        log.add(5000.0, dict(side='worker', rank=3, line=99, rows=dict(r=[])), parts=2, part=1)
        graded = self.grade(log.text())
        self.assertEqual(graded['torn']['lost_outside_every_arm'], 1)
        self.assertEqual(graded['arms']['fused-live-0']['torn']['lost_flushes'], 0)

    def test_torn_head_reads_only_before_the_rows(self):
        line = ('2026-11-01T20:06:28.347311206Z (Worker_PP0_TP1_EP1 pid=102) INFO flash-s5 progress '
                '{"base_ns":59,"line":63,"part":1,"parts":2,"rank":1,"rows":{"r":[[1,"line":7,')
        head = torn.torn_head(line, line.find(progress.PROGRESS_LINE))
        self.assertEqual((head['side'], head['rank'], head['line'], head['part'], head['parts']), ('worker', 1, 63, 1, 2))


if __name__ == '__main__':
    unittest.main()
