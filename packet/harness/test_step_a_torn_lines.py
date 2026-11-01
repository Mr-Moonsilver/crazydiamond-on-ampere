#!/usr/bin/env python3
"""The torn-line rule, on real-format lines through the real assemble and progress_receipts. Same rule as the C0 grader."""
import copy
import datetime
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
ARM = 'fused-live-0'
HEADER = '(Worker_PP1_TP1_EP1 pid=148) '
SIGNATURES = (('tp-ledger-mismatch', 'tp-ledger-mismatch'),)


class Log:
    """A run's engine log: for each chunk, an engine flush and four worker flushes. Rank 3 replies last."""

    def __init__(self, chunks=8, span=100.0):
        self.lines = []
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
                worker = [seq, 'flash_s5_submit', at + 1 + rank, at + 2 + rank, at + 3 + rank, 0, 1]
                self.add(stamp, dict(side='worker', rank=rank, line=index, rows=dict(r=[worker])))

    def add(self, stamp, body, parts=1, part=0):
        self.lines.append((stamp, dict(body, base_ns=0, dropped=0, fields=ROWS, part=part, parts=parts,
                                       schema=progress.PROGRESS_SCHEMA)))

    def text(self, tear=(), drop=(), splice=None, extra=()):
        """tear and drop hold (side, rank, line); splice maps one to text put inside its body."""
        out = []
        for stamp, body in self.lines:
            key = (body['side'], body['rank'], body['line'])
            if key in drop:
                continue
            who = '(Worker_PP0_TP0_EP0 pid=9) ' if body['side'] == 'worker' else '(EngineCore pid=1) '
            text = json.dumps(body, sort_keys=True, separators=(',', ':'))
            if key in tear:
                text = text[:-20]
            if splice and key in splice:
                text = text[:40] + splice[key] + text[40:]
            out.append('%s %sINFO [executor_hooks.py:1303] flash-s5 progress %s\n' % (when(stamp), who, text))
        return ''.join(out) + ''.join(extra)


def when(stamp):
    return datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f000Z')


def arm(label=ARM, chunks=8, span=100.0, first=0.0, cohort='live'):
    return dict(label=label, mode='fused', cohort=cohort, wall_start_s=999.0 + first, wall_end_s=1000.0 + first + span,
                progress_delta=dict(chunks=chunks), fresh=dict(prompt_tokens_observed=chunks * CHUNK + 1))


MEMORY = ('flash-s5 memory point=open rank=0 txn_seq=1 chunk=0-1024 allocated=1 reserved=2 inactive_split=3 '
          'alloc_retries=0 ooms=0 free=4')


class Rule(unittest.TestCase):
    def path(self, text):
        work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, work)
        path = os.path.join(work, 'engine-current.log')
        with open(path, 'w') as handle:
            handle.write(text)
        return path

    def grade(self, text, arms=None):
        return torn.receipts_for(arms or [arm()], [self.path(text)])[1]

    def row(self, text):
        return self.grade(text)['arms'][ARM]

    def test_a_clean_run_counts_nothing_and_the_recount_equals_the_joins_medians(self):
        path = self.path(Log().text())
        records, errors = progress.read_lines([path])
        base = progress.progress_receipts([arm()], records, errors)['arms'][ARM]
        row = torn.receipts_for([arm()], [path])[1]['arms'][ARM]
        self.assertEqual(row['tears'], dict(torn_progress_lines=0, torn_memory_lines=0, lost_flushes=[], unidentified_torn_lines=0,
                                            lost_records=0, dropped_chunk_starts=[], within_limits=True))
        for key in ('reasons', 'chunks', 'bands', 'all_counted', 'counted_chunks', 'missing_worker_rows'):
            self.assertEqual(row[key], base[key], key)
        again = copy.deepcopy(row)
        torn.recount(again)
        for key in ('bands', 'all_counted', 'counted_chunks'):
            self.assertEqual(again[key], row[key], key)

    def test_one_torn_worker_line_is_counted_and_its_chunk_drops_out_and_stays_listed(self):
        row = self.row(Log().text(tear={('worker', 3, 3)}))
        self.assertEqual(row['reasons'], [])
        self.assertEqual(row['tears']['torn_progress_lines'], 1)
        self.assertEqual(row['tears']['lost_records'], 1)
        self.assertEqual(row['tears']['dropped_chunk_starts'], [3 * CHUNK])
        self.assertTrue(row['tears']['within_limits'])
        self.assertEqual(len(row['chunks']), 8)
        self.assertEqual(row['counted_chunks'], 7)
        self.assertEqual(row['all_counted']['chunks'], 7)

    def test_a_flush_short_of_a_part_is_a_lost_flush(self):
        log = Log()
        log.add(1050.0, dict(side='worker', rank=3, line=99, rows=dict(r=[])), parts=2, part=1)
        graded = self.grade(log.text())
        self.assertEqual(graded['problems'], [])
        self.assertEqual(graded['arms'][ARM]['tears']['lost_flushes'],
                         [dict(side='worker', rank=3, line=99, evidence=['parts-incomplete'])])

    def test_two_lost_flushes_in_one_arm_go_inc_and_one_in_each_of_two_arms_does_not(self):
        row = self.row(Log().text(tear={('worker', 3, 2), ('worker', 3, 5)}))
        self.assertEqual(row['tears']['lost_records'], 2)
        self.assertFalse(row['tears']['within_limits'])
        self.assertEqual(row['reasons'], ['lost-records-above-the-declared-limit:2>1'])
        arms = [arm('a', chunks=4, span=40.0), arm('b', chunks=4, span=60.0, first=45.0)]
        arms[1]['fresh'] = dict(prompt_tokens_observed=8 * CHUNK + 1)
        graded = self.grade(Log().text(tear={('worker', 3, 1), ('worker', 3, 5)}), arms)
        self.assertEqual([graded['arms'][k]['tears']['lost_records'] for k in 'ab'], [1, 1])
        self.assertTrue(all(graded['arms'][k]['tears']['within_limits'] for k in 'ab'))

    def test_one_loss_touching_three_chunks_goes_inc_on_the_chunk_limit(self):
        log = Log()
        # One flush that held the critical rows of three chunks: its neighbours' ranges leave them all to it.
        log.lines = [(s, b) for s, b in log.lines if not (b['side'] == 'worker' and b['rank'] == 3 and b['line'] in (2, 3))]
        row = self.row(log.text(tear={('worker', 3, 4)}))
        self.assertEqual(row['tears']['dropped_chunk_starts'], [2 * CHUNK, 3 * CHUNK, 4 * CHUNK])
        self.assertIn('dropped-chunks-above-the-declared-limit:3>2', row['reasons'])

    def test_a_flush_gone_with_no_torn_text_is_found_by_its_writers_line_gap(self):
        row = self.row(Log().text(drop={('worker', 3, 3)}))
        self.assertEqual(row['tears']['lost_flushes'], [dict(side='worker', rank=3, line=3, evidence=['line-gap'])])
        self.assertEqual(row['tears']['dropped_chunk_starts'], [3 * CHUNK])
        self.assertEqual(row['reasons'], [])

    def test_a_missing_row_no_lost_flush_explains_still_fails_the_arm(self):
        log = Log()
        # Rank 3 holds every flush, without the row one chunk's engine line names.
        log.lines = [(s, dict(b, rows=dict(r=[]))) if (b['side'], b['rank'], b['line']) == ('worker', 3, 3) else (s, b)
                     for s, b in log.lines]
        row = self.row(log.text())
        self.assertIn('worker-rows-missing', row['reasons'])
        self.assertEqual(row['missing_worker_rows'], 1)
        self.assertEqual(row['tears']['lost_records'], 0)

    def test_a_lost_engine_flush_drops_its_chunk_and_clears_only_what_it_explains(self):
        row = self.row(Log().text(tear={('engine', None, 3)}))
        self.assertEqual(row['reasons'], [])
        self.assertEqual(row['tears']['lost_flushes'], [dict(side='engine', rank=None, line=3, evidence=['line-gap', 'torn-line'])])
        self.assertEqual(row['tears']['dropped_chunk_starts'], [3 * CHUNK])
        self.assertEqual(row['counted_chunks'], 7)
        before = next(c for c in row['chunks'] if c['chunk_end'] == 3 * CHUNK)
        after = next(c for c in row['chunks'] if c['chunk_start'] == 4 * CHUNK)
        self.assertNotIn('open_interval_s', before)
        self.assertNotIn('gap_before_s', after)

    def test_two_engine_chunks_lost_are_inc_on_the_record_limit(self):
        row = self.row(Log().text(tear={('engine', None, 3), ('engine', None, 5)}))
        self.assertIn('lost-records-above-the-declared-limit:2>1', row['reasons'])

    def test_a_torn_memory_line_is_counted_against_the_limit(self):
        stamp = when(1010.0)
        extra = ['%s (Worker_PP0_TP0_EP0 pid=9) INFO flash-s5 memory point=open rank=0 txn_seq=1 chunk=0-\n' % stamp]
        row = self.row(Log().text(extra=extra))
        self.assertEqual((row['tears']['torn_memory_lines'], row['tears']['lost_records']), (1, 1))
        self.assertTrue(row['tears']['within_limits'])
        row = self.row(Log().text(tear={('worker', 3, 2)}, extra=extra))
        self.assertEqual(row['tears']['lost_records'], 2)
        self.assertFalse(row['tears']['within_limits'])
        whole = ['%s (Worker_PP0_TP0_EP0 pid=9) INFO %s\n' % (stamp, MEMORY)]
        self.assertEqual(self.row(Log().text(extra=whole))['tears']['torn_memory_lines'], 0)

    def test_a_header_inside_a_json_string_still_parses_and_is_still_a_tear(self):
        log = Log()
        # A body whose JSON string holds another writer's header: it parses, so progress.read_lines keeps it.
        log.lines = [(s, dict(b, note=HEADER)) if (b['side'], b['rank'], b['line']) == ('worker', 3, 3) else (s, b)
                     for s, b in log.lines]
        text = log.text()
        records, errors = progress.read_lines([self.path(text)])
        self.assertEqual(errors, [])
        row = self.row(text)
        self.assertEqual(row['tears']['torn_progress_lines'], 1)
        self.assertEqual(row['tears']['lost_records'], 1)
        self.assertEqual(row['tears']['dropped_chunk_starts'], [3 * CHUNK])

    def test_a_header_alone_before_an_intact_body_is_not_a_tear(self):
        text = Log().text().replace('(EngineCore pid=1) INFO', '(EngineCore pid=1) (Worker_PP1_TP1_EP1 pid=148) INFO')
        row = self.row(text)
        self.assertEqual(row['tears']['torn_progress_lines'], 0)
        self.assertEqual(row['tears']['lost_records'], 0)
        self.assertEqual(row['reasons'], [])

    def test_a_line_off_schema_with_no_header_stays_a_problem(self):
        text = Log().text().replace('"schema":"%s"' % progress.PROGRESS_SCHEMA, '"schema":"other"', 1)
        self.assertIn('progress-line-schema', self.grade(text)['problems'])

    def test_a_signature_inside_a_torn_line_still_counts(self):
        line = '%s (Worker_PP0_TP0_EP0 pid=9) INFO flash-s5 progress {"base_ns":1,"li%%s\n' % when(1000.0)
        self.assertEqual(torn.signature_of(line % 'tp-ledger-mismatch', SIGNATURES), 'tp-ledger-mismatch')
        self.assertEqual(torn.signature_of(line % 'TP disagree on layer', SIGNATURES), 'tp-disagree-in-a-torn-line')
        split = '%s INFO native TP dis%sagree on layer\n' % (when(1000.0), HEADER)
        self.assertEqual(torn.signature_of(split, SIGNATURES), 'tp-disagree-in-a-torn-line')
        self.assertIsNone(torn.signature_of(line % '', SIGNATURES))
        whole = '%s (Worker_PP0_TP0_EP0 pid=9) INFO flash-s5 progress {"disagree":"tp-ledger-mismatch"}\n' % when(1000.0)
        self.assertIsNone(torn.signature_of(whole, SIGNATURES), 'a whole progress line holds no contract error')
        self.assertEqual(torn.signature_of('tp-ledger-mismatch in a plain line', SIGNATURES), 'tp-ledger-mismatch')
        self.assertIsNone(torn.signature_of('a plain line saying disagree', SIGNATURES), 'the plain scan owns this')

    def test_a_lost_flush_between_the_arms_is_reported_at_run_level_only(self):
        log = Log()
        log.add(5000.0, dict(side='worker', rank=3, line=99, rows=dict(r=[])), parts=2, part=1)
        graded = self.grade(log.text())
        self.assertEqual([x['line'] for x in graded['tears']['lost_flushes_outside_every_arm']], [99])
        self.assertEqual(graded['arms'][ARM]['tears']['lost_records'], 0)

    def test_an_unreadable_identity_counts_as_its_own_loss(self):
        lines = Log().text().splitlines(True)
        victim = next(i for i, x in enumerate(lines) if '"line":4,' in x and '"rank":3' in x)
        lines[victim] = lines[victim].split('flash-s5 progress ')[0] + 'flash-s5 progress {"base_ns":0,"dro\n'
        row = self.row(''.join(lines))
        self.assertEqual(row['tears']['unidentified_torn_lines'], 1)
        self.assertEqual(row['tears']['lost_records'], 2, 'the torn line, and the gap it leaves: errs toward INC')
        self.assertFalse(row['tears']['within_limits'])

    def test_the_limits_are_the_declared_ones(self):
        self.assertEqual((torn.LOST_RECORD_LIMIT, torn.DROPPED_CHUNK_LIMIT), (1, 2))


if __name__ == '__main__':
    unittest.main()
