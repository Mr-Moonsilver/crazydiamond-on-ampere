#!/usr/bin/env python3
"""Alpha-next PLAN section 2, the producer-side log fix: ReceiptWriter, emit sites, receipt_lines grader, CPU witness.

CPU only; no box, GPU or network. Leaves are declared in alpha_next_receipts_fixture_leaves.json.
`ALPHA_NEXT_RECEIPTS_ENGINE=<dir>` puts a mutated native_measurement.py first on the path (the mutant runner,
test_alpha_next_receipts_mutants.py); unset, the committed engine source is tested.
"""
from collections import deque
from contextlib import ExitStack
import ast
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import select
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'patches' / 's5_engine'
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT / 'harness'))

import alpha_next_receipts_witness as witness  # noqa: E402
import receipt_lines as rl  # noqa: E402

MUTANT_ENGINE = os.environ.get('ALPHA_NEXT_RECEIPTS_ENGINE')
if MUTANT_ENGINE:
    sys.path.insert(0, MUTANT_ENGINE)

import native_measurement as nm  # noqa: E402

LIMIT = 3968  # the plan's record budget including the newline; held here, not read from the module under test


def lines_from(writes):
    """Physical lines of the captured fd writes, in order."""
    return b''.join(data for _, data in writes).decode().splitlines(keepends=True)


class Capture:
    """A recording os.write leaf: (fd, bytes) per call; ``short``/``fail`` script one call."""

    def __init__(self, short_at=None, fail_at=None):
        self.writes, self.short_at, self.fail_at = [], short_at, fail_at

    def __call__(self, fd, data):
        index = len(self.writes)
        if index == self.fail_at:
            self.writes.append((fd, b''))
            raise OSError(32, 'broken pipe')
        if index == self.short_at:
            self.writes.append((fd, data[:len(data) // 2]))
            return len(data) // 2
        self.writes.append((fd, bytes(data)))
        return len(data)


def capture_writer(pid=4242, **kwargs):
    cap = Capture(**kwargs)
    r, w = os.pipe()
    writer = nm.ReceiptWriter(w, write=cap, getpid=lambda: pid)
    writer._pipe = (r, w)
    return writer, cap


def close_pipe(writer):
    for fd in writer._pipe:
        os.close(fd)


def progress_record(rows=600, side='worker', rank=1, **identity):
    trace = nm.ProgressTrace(side, rank)
    for i in range(rows):
        t = 10 ** 15 + i * 1000
        trace.record('r', i, 'execute_model', t, t + 900, t + 990, False, True)
    return trace.lines(txn=identity.get('txn', 7), chunk_start=0, chunk_end=1024)


ALLOC = ('flash-s5 native-alloc rank=1 txn_seq=7 chunk=0-1024 union_retire=1 union_sets=2 union_retired=1 '
         'union_single=10 union_live_max=10 indexer_layers=1 indexer_rows=1024 indexer_width=12544 indexer_chunks=1 '
         'indexer_peak=1 indexer_total=1 indexer_errors=0')
MEMORY = ('flash-s5 memory point=open rank=1 txn_seq=7 chunk=0-1024 allocated=1 reserved=2 inactive_split=3 '
          'alloc_retries=0 ooms=0 free=5 total=6 sync_all_streams=0 device_allocs=1 device_frees=1')
STAGE = ('flash-s5 stage-local txn=7 chunk=0-1024 commands=3,4 both_stages_ns=9 first_ns=1,2 last_ns=3,4 '
         'local_release_ns=5,6 headroom_refused=0 headroom_blocked_ns=0')


class Budget(unittest.TestCase):
    def test_record_budget_is_below_pipe_buf_and_everyone_follows_it(self):
        import handler_anatomy
        self.assertEqual(nm.RECEIPT_RECORD_BYTES, LIMIT)
        self.assertEqual(nm.PROGRESS_LINE_BYTES, LIMIT)
        self.assertLessEqual(LIMIT, select.PIPE_BUF - 128)
        self.assertEqual(handler_anatomy.LINE_BYTES, nm.PROGRESS_LINE_BYTES)
        self.assertEqual(handler_anatomy.IDENTITY_BYTES, nm.RECEIPT_IDENTITY_BYTES)
        with self.assertRaises(ValueError):
            nm.ReceiptWriter(1, limit=LIMIT + 1)

    def test_a_full_progress_trace_splits_into_records_below_the_budget(self):
        writer, cap = capture_writer()
        trace = nm.ProgressTrace('engine', limit=nm.PROGRESS_MAX_ROWS)
        for seq in range(nm.PROGRESS_MAX_ROWS):
            t = (1 << 62) + seq * 10 ** 9
            trace.record('n', 10 ** 18 + seq, 'segment' * 3, 'state_finalize', 10 ** 18, seq % 2, 47, 'tick-owner',
                         t, t + 1, t + 2, t + 3, t + 4, t + 5, t + 6, 'accepted-after-pending', 1)
        texts = trace.lines(txn=10 ** 12, chunk_start=262143 * 1024, chunk_end=262144 * 1024, mode='burst',
                            open_rows=10 ** 6)
        self.assertGreater(len(texts), 50)
        self.assertTrue(writer.write_lines(texts))
        sizes = [len(data) for _, data in cap.writes]
        self.assertLessEqual(max(sizes), LIMIT)
        self.assertEqual((writer.oversize, writer.short_writes, writer.write_errors, writer.faults), (0, 0, 0, 0))
        self.assertEqual(len(cap.writes), len(texts) + 1)
        report = rl.check(lines_from(cap.writes))
        self.assertEqual(rl.verdict(report)['verdict'], 'OK', report)
        close_pipe(writer)

    def test_anatomy_trace_with_every_tag_stays_below_the_budget(self):
        writer, cap = capture_writer()
        trace = nm.AnatomyTrace('worker', 3)
        big = 10 ** 15
        for i in range(2000):
            tag = 'ckqfxep'[i % 7]
            fields = [big + i] * len(nm.ANATOMY_ROWS[tag])
            trace.record(tag, *fields)
        texts = trace.lines(txn=1, request_generation=2, chunk_start=0, chunk_end=1024, stage=1)
        self.assertTrue(writer.write_lines(texts))
        self.assertLessEqual(max(len(d) for _, d in cap.writes), LIMIT)
        self.assertEqual(writer.oversize, 0)
        self.assertEqual(rl.verdict(rl.check(lines_from(cap.writes)))['verdict'], 'OK')
        close_pipe(writer)


class Writer(unittest.TestCase):
    def test_one_write_per_part_with_identity_on_every_part(self):
        writer, cap = capture_writer(pid=77)
        texts = progress_record()
        self.assertGreater(len(texts), 2)
        self.assertTrue(writer.write_lines(texts))
        self.assertEqual(len(cap.writes), 1 + len(texts))          # path record, then one write per part
        path = cap.writes[0][1].decode()
        self.assertTrue(path.startswith('flash-s5 receipts receipt_path=atomic-fifo pipe_buf='), path)
        self.assertIn(' pid=77 seq=0 sha=', path)
        sha = nm.receipt_sha(texts)
        self.assertEqual(sha, hashlib.sha256('\n'.join(texts).encode()).hexdigest()[:16])
        for index, (fd, data) in enumerate(cap.writes[1:]):
            self.assertEqual(fd, writer._pipe[1])
            text = data.decode()
            self.assertTrue(text.endswith('\n') and text.count('\n') == 1)
            self.assertTrue(text.startswith(nm.PROGRESS_LINE + '{"pid":77,"seq":%d,"sha":"%s",' % (index + 1, sha)))
            body = json.loads(text[len(nm.PROGRESS_LINE):])
            self.assertEqual((body['part'], body['parts']), (index, len(texts)))
            self.assertLessEqual(len(data), LIMIT)
        self.assertEqual(writer.stats()['records'], 1 + len(texts))
        close_pipe(writer)

    def test_seq_is_strictly_plus_one_across_every_record_kind(self):
        writer, cap = capture_writer()
        anatomy = nm.AnatomyTrace('worker', 1)
        anatomy.record('f', 'credit', 3, 9)
        handler = 'flash-s5 handler ' + json.dumps(dict(schema='s5-handler-anatomy/1', rank=1, line=0, base_ns=0,
                                                        part=0, parts=1, rows={}), separators=(',', ':'))
        for _ in range(3):
            self.assertTrue(writer.write_kv(ALLOC))
            self.assertTrue(writer.write_lines(progress_record(rows=200)))
            self.assertTrue(writer.write_kv(MEMORY))
            self.assertTrue(writer.write_lines(anatomy.lines(txn=1)))
            anatomy.record('f', 'credit', 3, 9)
            self.assertTrue(writer.write_kv(STAGE))
            self.assertTrue(writer.write_lines([handler]))
        lines = lines_from(cap.writes)
        seqs = [int(re.search(r'(?:"seq":| seq=)(\d+)', line).group(1)) for line in lines]
        self.assertEqual(seqs, list(range(len(lines))))
        report = rl.check(lines)
        self.assertEqual(rl.verdict(report)['verdict'], 'OK', report)
        kinds = report['per_process']['4242']['kinds']
        self.assertEqual(set(kinds), {'receipts', 'native-alloc', 'progress', 'memory', 'anatomy', 'stage-local',
                                      'handler'})
        close_pipe(writer)

    def test_later_kinds_stop_and_memory_peak_are_records_too(self):
        writer, cap = capture_writer()
        stop = 'flash-s5 stop ' + json.dumps(dict(schema='s5-stop-receipt/1', side='engine', kind='stop_partial_replay',
                                                  txn=3, op_seq=9), separators=(',', ':'), sort_keys=True)
        peak = 'flash-s5 memory-peak rank=2 txn_seq=3 chunk=0-1024 peak=7'
        self.assertTrue(writer.write_lines([stop]))
        self.assertTrue(writer.write_kv(peak))
        lines = lines_from(cap.writes)
        report = rl.check(lines)
        self.assertEqual(rl.verdict(report)['verdict'], 'OK', report)
        self.assertEqual(report['per_process']['4242']['kinds'], {'receipts': 1, 'stop': 1, 'memory-peak': 1})
        forged = lines[1].replace('"op_seq":9', '"op_seq":8')
        self.assertEqual(len(rl.check([lines[0], forged, lines[2]])['bad_sha']), 1)
        close_pipe(writer)

    def test_kv_records_keep_their_text_and_gain_pid_seq_sha(self):
        writer, cap = capture_writer(pid=9)
        writer.write_kv(MEMORY)
        text = cap.writes[1][1].decode()
        self.assertEqual(text, '%s pid=9 seq=1 sha=%s\n' % (MEMORY, nm.receipt_sha([MEMORY])))
        close_pipe(writer)

    def test_short_write_is_counted_and_never_raised(self):
        writer, cap = capture_writer(short_at=2)
        self.assertTrue(writer.write_kv(ALLOC))
        self.assertFalse(writer.write_kv(MEMORY))
        self.assertEqual((writer.short_writes, writer.write_errors, writer.records), (1, 0, 2))
        self.assertTrue(writer.write_kv(STAGE))                     # numbering goes on: the loss shows as a gap
        self.assertEqual(writer.seq, 4)
        self.assertEqual(len(cap.writes), 4)                        # no retry of the rest
        report = rl.check(lines_from(cap.writes))
        self.assertTrue(report['missing_seq'] or report['bad_sha'], report)
        self.assertEqual(rl.verdict(report)['verdict'], 'INC')
        close_pipe(writer)

    def test_failed_write_is_counted_and_never_raised(self):
        writer, cap = capture_writer(fail_at=1)
        self.assertFalse(writer.write_lines(progress_record(rows=10)))
        self.assertEqual((writer.write_errors, writer.short_writes), (1, 0))
        self.assertTrue(writer.write_kv(MEMORY))
        self.assertEqual(rl.verdict(rl.check(lines_from(cap.writes)))['verdict'], 'INC')
        close_pipe(writer)

    def test_oversize_record_is_counted_not_written_and_shows_as_a_gap(self):
        writer, cap = capture_writer()
        self.assertFalse(writer.write_kv(MEMORY + ' pad=' + 'x' * LIMIT))
        self.assertEqual((writer.oversize, len(cap.writes)), (1, 1))
        writer.write_kv(MEMORY)
        report = rl.check(lines_from(cap.writes))
        self.assertEqual(report['missing_seq'], [dict(pid=4242, after=0, got=2)])
        close_pipe(writer)

    def test_foreign_or_malformed_text_is_counted_and_numbered(self):
        writer, cap = capture_writer()
        self.assertFalse(writer.write_kv('not a receipt'))
        self.assertFalse(writer.write_kv(MEMORY + '\nflash-s5 memory forged'))
        self.assertFalse(writer.write_lines(['flash-s5 progress not-json']))
        self.assertEqual(writer.faults, 3)
        self.assertEqual(writer.seq, 4)
        close_pipe(writer)

    def test_nothing_is_opened_or_written_until_a_receipt(self):
        cap = Capture()
        writer = nm.ReceiptWriter(write=cap)
        with mock.patch.object(nm, '_RECEIPT_WRITER', None):
            self.assertIs(nm.install_receipt_writer(writer), writer)
            self.assertIs(nm.install_receipt_writer(), writer)       # idempotent
            self.assertIs(nm.installed_receipt_writer(), writer)
        self.assertEqual((writer.fd, writer.pid, cap.writes), (None, None, []))

    def test_private_dup_of_fd_1_on_first_write(self):
        cap = Capture()
        writer = nm.ReceiptWriter(write=cap)
        writer.write_kv(MEMORY)
        try:
            self.assertNotIn(writer.fd, (None, 0, 1, 2))
            self.assertTrue(os.path.sameopenfile(writer.fd, 1))
            self.assertFalse(os.get_inheritable(writer.fd))
        finally:
            os.close(writer.fd)

    def test_a_forked_child_starts_its_own_numbering(self):
        pids = [100]
        cap = Capture()
        r, w = os.pipe()
        writer = nm.ReceiptWriter(w, write=cap, getpid=lambda: pids[0])
        writer.write_kv(MEMORY)
        pids[0] = 101
        writer.write_kv(MEMORY)
        lines = lines_from(cap.writes)
        self.assertEqual([re.search(r' pid=(\d+) seq=(\d+)', x).groups() for x in lines],
                         [('100', '0'), ('100', '1'), ('101', '0'), ('101', '1')])
        self.assertEqual(rl.verdict(rl.check(lines))['verdict'], 'OK')
        os.close(r)
        os.close(w)

    def test_uninstalled_process_keeps_the_old_logger_lines_and_the_grader_calls_them_legacy(self):
        logger = mock.Mock()
        texts = progress_record(rows=10)
        with mock.patch.object(nm, '_RECEIPT_WRITER', None):
            nm.emit_receipt_lines(texts, logger)
            nm.emit_receipt_kv(MEMORY, logger)
        self.assertEqual([c.args for c in logger.info.call_args_list], [(t,) for t in texts] + [(MEMORY,)])
        report = rl.check(['2026-11-02T00:00:00Z (Worker_PP0_TP1_EP1 pid=5) INFO x %s\n' % t for t in texts + [MEMORY]])
        self.assertEqual([u['reason'] for u in report['unsafe_path']], ['legacy:memory', 'legacy:progress'])
        self.assertEqual(rl.verdict(report)['verdict'], 'INC')


class ReceiptPath(unittest.TestCase):
    def test_pipe_is_atomic_fifo_and_a_file_is_unsafe(self):
        r, w = os.pipe()
        try:
            self.assertEqual(nm.receipt_path(w), 'atomic-fifo pipe_buf=%d' % select.PIPE_BUF)
        finally:
            os.close(r)
            os.close(w)
        with tempfile.TemporaryFile() as handle:
            self.assertEqual(nm.receipt_path(handle.fileno()), 'unsafe:not-fifo')
            writer = nm.ReceiptWriter(handle.fileno(), getpid=lambda: 5)
            self.assertTrue(writer.write_kv(MEMORY))                 # still written; reported unsafe
            handle.seek(0)
            lines = handle.read().decode().splitlines(keepends=True)
        report = rl.check(lines)
        self.assertEqual(report['unsafe_path'], [dict(pid=5, reason='unsafe:not-fifo')])
        self.assertEqual(rl.verdict(report), dict(verdict='INC', inc_processes={'5': ['unsafe_path']}, log_wide=[]))

    def test_small_pipe_buf_or_closed_fd_is_unsafe(self):
        r, w = os.pipe()
        try:
            with mock.patch.object(select, 'PIPE_BUF', 512):
                self.assertEqual(nm.receipt_path(w), 'unsafe:pipe-buf-512')
        finally:
            os.close(r)
            os.close(w)
        self.assertTrue(nm.receipt_path(w).startswith('unsafe:fstat-'))

    def test_startup_line_is_read_and_must_be_atomic(self):
        ok = '(EngineCore pid=1) INFO [s5_startup.py:400] flash-s5 startup receipt_path=atomic-fifo pipe_buf=4096\n'
        bad = '(EngineCore pid=1) INFO [s5_startup.py:400] flash-s5 startup receipt_path=unsafe:not-fifo\n'
        self.assertEqual(rl.check([ok], require_startup=True)['unsafe_path'], [])
        self.assertEqual(rl.check([bad])['unsafe_path'], [dict(pid=None, reason='startup:unsafe:not-fifo')])
        self.assertEqual(rl.check([], require_startup=True)['unsafe_path'], [dict(pid=None, reason='no-startup-line')])


class Grader(unittest.TestCase):
    def stream(self, *records, pid=4242):
        writer, cap = capture_writer(pid=pid)
        for kind, payload in records:
            (writer.write_lines if kind == 'lines' else writer.write_kv)(payload)
        close_pipe(writer)
        return lines_from(cap.writes)

    def test_clean_stream_with_foreign_bytes_before_a_record_is_ok(self):
        lines = self.stream(('lines', progress_record()), ('kv', MEMORY))
        lines[2] = '(Worker_PP9_TP9_EP9 pid=3) INFO half a vllm line without its newline ' + lines[2]
        lines.insert(1, '(Worker_PP9_TP9_EP9 pid=3) INFO an ordinary line\n')
        report = rl.check(lines)
        self.assertEqual(rl.verdict(report)['verdict'], 'OK', report)

    def test_a_lost_part_is_incomplete_and_a_seq_gap(self):
        lines = self.stream(('lines', progress_record()), ('kv', MEMORY))
        del lines[2]
        report = rl.check(lines)
        self.assertEqual(len(report['incomplete']), 1)
        self.assertEqual(report['missing_seq'], [dict(pid=4242, after=1, got=3)])

    def test_a_repeated_or_reordered_seq_is_missing_seq(self):
        lines = self.stream(('kv', ALLOC), ('kv', MEMORY), ('kv', STAGE))
        self.assertEqual(rl.check(lines[:2] + lines[1:])['missing_seq'], [dict(pid=4242, after=1, got=1)])
        self.assertEqual(len(rl.check([lines[0], lines[2], lines[1], lines[3]])['missing_seq']), 3)

    def test_a_spliced_record_is_unreadable_or_bad_sha(self):
        lines = self.stream(('lines', progress_record()), ('kv', MEMORY))
        torn = lines[1][:500] + 'XXXX' + lines[1][500:]
        report = rl.check([lines[0], torn] + lines[2:])
        self.assertTrue(report['missing_seq'] or report['bad_sha'], report)
        value = lines[1].replace('"execute_model"', '"execute_mode1"', 1)
        report = rl.check([lines[0], value] + lines[2:])
        self.assertEqual(len(report['bad_sha']), 1)
        kv = lines[-1].replace('allocated=1', 'allocated=7')
        self.assertEqual(len(rl.check(lines[:-1] + [kv])['bad_sha']), 1)

    def test_per_part_sha_is_bad_sha(self):
        texts = progress_record()
        lines = self.stream(('lines', texts))
        forged = []
        for line in lines[1:]:
            _, body = rl._record_of(line)
            m = rl.JSON_IDENTITY.match(body[len(nm.PROGRESS_LINE):])
            bare = nm.PROGRESS_LINE + '{' + body[len(nm.PROGRESS_LINE) + m.end():]
            forged.append(line.replace(m.group(3), nm.receipt_sha([bare])))
        self.assertEqual(len(rl.check(lines[:1] + forged)['bad_sha']), 1)

    def test_a_window_without_the_process_start(self):
        lines = self.stream(('kv', ALLOC), ('kv', MEMORY))
        self.assertTrue(rl.check(lines[1:])['missing_seq'])
        self.assertEqual(rl.verdict(rl.check(lines[1:], require_start=False))['verdict'], 'OK')

    def test_cli_reads_xz_and_exits_nonzero_on_inc(self):
        import lzma
        lines = self.stream(('lines', progress_record()), ('kv', MEMORY))
        with tempfile.TemporaryDirectory() as d:
            good, bad = Path(d) / 'good.log.xz', Path(d) / 'bad.log'
            with lzma.open(good, 'wt') as handle:
                handle.write(''.join(lines))
            bad.write_text(''.join(lines[:2] + lines[3:]))
            out = io.StringIO()
            with mock.patch('sys.stdout', out):
                self.assertEqual(rl.main([str(good)]), 0)
                self.assertEqual(rl.main([str(bad)]), 1)


class ExistingParsers(unittest.TestCase):
    """Prefixes unchanged: the frozen parsers still read every new record (they ignore pid/seq/sha)."""

    def records(self):
        writer, cap = capture_writer()
        texts = progress_record(rows=900, side='worker', rank=2)
        writer.write_lines(texts)
        writer.write_kv(MEMORY)
        writer.write_kv(ALLOC)
        close_pipe(writer)
        return texts, ['2026-11-02T01:00:%02d.5Z %s' % (i % 60, x) for i, x in enumerate(lines_from(cap.writes))]

    def test_progress_box_reads_and_assembles_every_part(self):
        import e3_native41_progress_box as box
        texts, lines = self.records()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'engine-stream.log'
            path.write_text(''.join(lines))
            records, errors = box.read_lines([path])
        self.assertEqual(errors, [])
        self.assertEqual(len(records), len(texts))
        flushes, problems = box.assemble(records)
        self.assertEqual(problems, [])
        self.assertEqual(len(flushes), 1)
        self.assertEqual(len(flushes[0]['rows']['r']), 900)

    def test_torn_rules_and_memory_regexes_read_them_whole(self):
        import step_a_torn_lines as torn
        import native43_allocator_phase_deltas as phases
        _, lines = self.records()
        self.assertFalse(any(torn.is_torn_progress(x) for x in lines))
        self.assertFalse(any(torn.is_torn_memory(x) for x in lines))
        memory = [x for x in lines if 'flash-s5 memory ' in x]
        self.assertTrue(torn.MEMORY_LINE.search(memory[0]))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'engine.log'
            path.write_text(''.join(lines))
            mem, alloc, count = phases.read_rows([path])
        self.assertEqual((count, sorted(mem), sorted(alloc)), (2, [1], [1]))
        self.assertEqual(mem[1][0]['alloc_retries'], 0)


class EmitSites(unittest.TestCase):
    """Every receipt emit site goes through the one installed writer; nothing goes to the logger."""

    def setUp(self):
        self.writer, self.cap = capture_writer()
        patcher = mock.patch.object(nm, '_RECEIPT_WRITER', self.writer)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(close_pipe, self.writer)
        self.logger = mock.Mock()

    def kinds(self):
        lines = lines_from(self.cap.writes)
        report = rl.check(lines)
        self.assertEqual(rl.verdict(report)['verdict'], 'OK', report)
        self.assertLessEqual(max(len(d) for _, d in self.cap.writes), LIMIT)
        return report['per_process']['4242']['kinds']

    def test_engine_progress_and_step_anatomy(self):
        import engine_driver as ed
        progress = ed.NativeProgress('trace', types.SimpleNamespace())
        progress.begin(dict(txn=3, chunk_start=0, chunk_end=1024))
        for i in range(700):
            progress.trace.record('s', i, 'sample_tokens', 10 ** 15 + i, 10 ** 15 + i + 5, 10 ** 15 + i + 9)
        progress.end(self.logger)
        self.logger.info.assert_not_called()
        kinds = self.kinds()
        self.assertGreater(kinds['progress'], 1)
        self.assertEqual(kinds['anatomy'], 1)
        self.assertEqual(progress.counters['line_errors'], 0)

    def test_worker_progress_at_release(self):
        import executor_hooks as hooks
        trace = nm.ProgressTrace('worker', 1)
        for i in range(500):
            trace.record('r', i, 'execute_model', 10 ** 15, 10 ** 15 + 1, 10 ** 15 + 2, False, True)
        proc = types.SimpleNamespace(rank=1)
        with mock.patch.object(hooks, '_WORKER_PROGRESS', (trace, deque())), \
                mock.patch.object(hooks, '_ANATOMY', False), \
                mock.patch.object(hooks, '_progress_log', self.logger):
            hooks.worker_rpc_begin(proc, 'flash_s5_release', {nm.PROGRESS_META_KEY: nm.encode_progress_meta(5, False)})
        self.logger.info.assert_not_called()
        self.assertGreater(self.kinds()['progress'], 1)
        self.assertEqual(trace.dropped_total, 0)

    def test_qsa_native_alloc(self):
        import qsa_operation as qo
        stats = dict(union_scratch_sets=6, union_scratch_retired=5, union_scratch_bytes=1, union_scratch_live_max_bytes=2,
                     indexer_layers=6, indexer_rows=6144, indexer_width=12544, indexer_chunks=6, indexer_peak_bytes=3,
                     indexer_total_bytes=4, indexer_errors=0, alloc_receipt_errors=0)
        op = types.SimpleNamespace(stats=stats, transaction=types.SimpleNamespace(txn_seq=1, chunk_start=0, chunk_end=1024),
                                   lane=types.SimpleNamespace(worker_rank=2), union_retire=True)
        with self.assertNoLogs('qsa_operation', level='INFO'):
            qo.QsaPrefillOperation._alloc_receipt(op)
        self.assertEqual(stats['alloc_receipt_errors'], 0)
        self.assertEqual(self.kinds()['native-alloc'], 1)
        line = lines_from(self.cap.writes)[1]
        self.assertTrue(line.startswith('flash-s5 native-alloc rank=2 txn_seq=1 chunk=0-1024 union_retire=1 '), line)

    def test_worker_bridge_memory_peak(self):
        import worker_bridge
        import test_alpha_next_peak_receipt as pk
        cuda = pk.PeakCuda()
        bridge = pk.bridge(cuda)
        worker_bridge.NativeWorkerBridge._peak_open(bridge)
        cuda.set(5, 6)
        with self.assertNoLogs('worker_bridge', level='INFO'):
            worker_bridge.NativeWorkerBridge._peak_receipt(bridge, pk.TX)
        self.assertEqual(bridge.peak_receipt_errors, 0)
        self.assertEqual(self.kinds()['memory-peak'], 1)
        self.assertIsNotNone(pk.LINE.search(rl.parse_line(lines_from(self.cap.writes)[1])['text']))

    def test_engine_stop_receipt(self):
        import engine_driver as ed
        driver = types.SimpleNamespace(progress=types.SimpleNamespace(counters={'line_errors': 0}),
                                       _monotonic_ns=lambda: 10 ** 15)
        with mock.patch.object(ed, '_progress_logger', lambda: self.logger):
            ed.NativeSegmentDriver._stop_receipt(driver, 'stop_partial_replay', dict(transaction=4, op_seq=9))
        self.logger.info.assert_not_called()
        self.assertEqual(driver.progress.counters['line_errors'], 0)
        self.assertEqual(self.kinds()['stop'], 1)
        line = lines_from(self.cap.writes)[1]
        self.assertTrue(line.startswith(ed.STOP_RECEIPT_LINE), line)

    def test_worker_bridge_memory(self):
        import worker_bridge
        import test_alpha_m3_receipts as m3
        bridge = types.SimpleNamespace(rank=3, _memory_cuda=m3.FakeCuda(), memory_rows=[], memory_receipt_errors=0)
        with self.assertNoLogs('worker_bridge', level='INFO'):
            worker_bridge.NativeWorkerBridge._memory_receipt(bridge, 'open',
                                                             types.SimpleNamespace(txn_seq=1, chunk_start=0, chunk_end=1024))
        self.assertEqual(bridge.memory_receipt_errors, 0)
        self.assertEqual(self.kinds()['memory'], 1)
        self.assertTrue(lines_from(self.cap.writes)[1].startswith('flash-s5 memory point=open rank=3 txn_seq=1 '))

    def test_c0_stage_local_line(self):
        import stage_progress
        driver = types.SimpleNamespace(progress=object(), _headroom_chunk=0, _ring_blocked_chunk_ns=0)
        stats = dict(commands=[3, 4], first_ns=[1, 2], last_ns=[3, 4], local_release_ns=[5, 6])
        with mock.patch.object(stage_progress, '_progress_logger', lambda: self.logger):
            stage_progress.StagedSegmentDriver._stage_line(
                driver, types.SimpleNamespace(txn_seq=7, chunk_start=0, chunk_end=1024), stats, 9)
        self.logger.info.assert_not_called()
        self.assertEqual(self.kinds()['stage-local'], 1)
        self.assertEqual(rl.parse_line(lines_from(self.cap.writes)[1])['text'], STAGE)

    def test_device_spans_send_one_chunk_as_one_record(self):
        import test_e3_native_anatomy as an
        clock, log, single, groups = an.FakeClock(), [], [], []
        spans = nm.DeviceSpans(lambda: an.TimingEvent(clock, log), rank=1, emit=single.append, clock=clock,
                               emit_lines=lambda texts: (groups.append(list(texts)),
                                                         nm.emit_receipt_lines(texts, self.logger)))
        spans.begin(an.TXN, 0)
        for op_seq in range(200):
            mark = spans.start('prefill', None)
            spans.end(mark, 'prefill', an.command('prepare', op_seq), None)
        self.assertTrue(spans.release('prefill', None))
        self.assertEqual(single, [])
        self.assertEqual(len(groups), 1)
        self.assertGreater(len(groups[0]), 1)
        self.assertEqual(spans.lines_emitted, len(groups[0]))
        self.assertEqual(self.kinds()['anatomy'], len(groups[0]))

    def test_startup_device_spans_route_through_the_writer(self):
        import s5_startup
        torch = types.ModuleType('torch')
        torch.cuda = types.SimpleNamespace(Event=lambda enable_timing: object())
        with mock.patch.dict(sys.modules, {'torch': torch}):
            spans = s5_startup._device_spans(1)
        spans.emit_lines(progress_record(rows=10))
        self.assertEqual(self.kinds()['progress'], 1)

    def test_handler_anatomy_flush_on_the_stage1_rig(self):
        import test_stage1_handler_anatomy as s1
        with ExitStack() as stack:
            rig = s1.Rig(stack, 2)
            for _ in range(120):
                rig.step()
            rig.replies(240)
            rig.release()
            self.assertEqual(rig.handler_lines(), [])           # nothing through the logger
            self.assertEqual(rig.progress_lines(), [])
        kinds = self.kinds()
        self.assertGreater(kinds['handler'], 1)
        self.assertGreater(kinds['progress'], 1)
        handler = [rl._record_of(x)[1] for x in lines_from(self.cap.writes) if 'flash-s5 handler ' in x]
        heads, main, out = s1.parse(handler)
        self.assertEqual(len(heads), 1)
        self.assertEqual(len(main), 240)

    def test_startup_installs_the_writer_in_engine_and_worker_and_reports_the_path(self):
        tree = ast.parse((ENGINE / 's5_startup.py').read_text())
        funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

        def calls(name):
            return [n.func.id for n in ast.walk(funcs[name]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        engine, worker = calls('install_engine'), calls('install_native_lane')
        self.assertIn('install_receipt_writer', worker)
        self.assertLess(worker.index('install_receipt_writer'), worker.index('LaneDispatcher'))
        self.assertLess(engine.index('install_receipt_writer'), engine.index('bootstrap_full_startup'))
        self.assertIn('receipt_path', engine)
        source = (ENGINE / 's5_startup.py').read_text()
        self.assertIn("'flash-s5 startup receipt_path=%s', receipt_path()", source)


class Witness(unittest.TestCase):
    """4 writers + 1 noise process on one os.pipe. New path: 0 bad, contiguous seq (smaller than the logged 1e5 run)."""
    RECORDS = 20_000

    def test_new_path_has_no_bad_record_and_contiguous_seq(self):
        result = witness.run('new', writers=4, records=self.RECORDS, seed=7)
        self.assertGreaterEqual(result['records_read'], self.RECORDS)
        self.assertEqual((result['missing_seq'], result['bad_sha'], result['incomplete']), (0, 0, 0), result)
        self.assertEqual(result['unsafe_path'], [])
        self.assertEqual(len(result['per_process']), 4)
        for proc in result['per_process'].values():
            self.assertEqual(proc['first_seq'], 0)
            self.assertEqual(proc['last_seq'] + 1, proc['records'])

    def test_old_path_leaves_a_long_record_in_several_writes(self):
        """Structural half of the witness: through the decorated buffered stream one 12000-byte record is >1 write(2)."""
        r, w = os.pipe()
        blk = os.fstat(w).st_blksize
        os.close(r)
        os.close(w)

        class Raw(io.RawIOBase):
            calls = []

            def writable(self):
                return True

            def write(self, data):
                self.calls.append(len(data))
                return len(data)
        stream = io.TextIOWrapper(io.BufferedWriter(Raw(), buffer_size=blk))
        logger = witness.old_path_logger('Worker_PP0_TP1_EP1', stream=stream)
        logger.info('flash-s5 progress ' + 'x' * 11900)
        self.assertGreater(len(Raw.calls), 1, Raw.calls)
        self.assertGreater(sum(Raw.calls), select.PIPE_BUF)

    def test_fixture_leaf_is_the_declared_copy(self):
        import inspect
        leaves = json.loads((ROOT / 'harness' / 'alpha_next_receipts_fixture_leaves.json').read_text())
        leaf, = [x for x in leaves['leaves'] if x['name'] == 'alpha-next-receipts-vllm-add-prefix']
        source = inspect.getsource(witness.vllm_add_prefix)
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), leaf['leaf_source_sha256'])
        stream = io.StringIO()
        witness.vllm_add_prefix(stream, 'Worker_PP0_TP0_EP0', 9)
        stream.write('a\nb')
        stream.write('c\n')
        self.assertEqual(stream.getvalue(), '(Worker_PP0_TP0_EP0 pid=9) a\n(Worker_PP0_TP0_EP0 pid=9) bc\n')


if __name__ == '__main__':
    if MUTANT_ENGINE:
        print('native_measurement under test: ' + nm.__file__, file=sys.stderr)
    unittest.main()
