#!/usr/bin/env python3
"""C1 report (tools/alpha_c1_report.py): regrouping interleaved chunks by the stage-overlap lines, the slower TP rank per stage,
and the commit interval. Synthetic rows in the box record's shape, and (RealLines) the real overlap driver, worker hooks and
emitters on the per-rank ring with a slow stage 1, their lines through the grade's own readers and the regroup. CPU only."""
import collections
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
import alpha_c1_report as r  # noqa: E402


def n_row(seq, stage, sent, recv, applied, kind='segment'):
    row = dict(seq=seq, kind=kind, segment='layer', op_seq=seq, stage=stage, layer=0, origin='tick', t_submit=sent - 10,
               t_sent=sent, t_apply=applied, outcome='accepted', direct=1)
    for rank in range(4):
        row['t_recv%d' % rank] = recv.get(rank)
    return row


class Regroup(unittest.TestCase):
    def flushes(self):
        # Flush of txn 1 holds txn 2's early stage-0 rows (seqs 5, 6); flush of txn 2 holds the rest of txn 2.
        one = dict(side='engine', line=1, txn=1, identity=dict(chunk_start=0, chunk_end=4, decode_live=8, txn=1),
                   rows={'n': [n_row(1, 0, 100, {0: 200, 1: 260}, 270), n_row(2, 1, 300, {2: 400, 3: 410}, 420),
                               n_row(5, 0, 310, {0: 380, 1: 350}, 390), n_row(6, 0, 400, {0: 450, 1: 500}, 510)],
                         's': [dict(seq=3, kind='execute_model', t_submit=150, t_recv=160, t_take=170)]})
        two = dict(side='engine', line=2, txn=2, identity=dict(chunk_start=4, chunk_end=8, decode_live=8, txn=2),
                   rows={'n': [n_row(7, 1, 520, {2: 600, 3: 700}, 710)]})
        return [one, two]

    def test_rows_follow_their_chunk_and_stock_rows_their_flush(self):
        texts = ['x flash-s5 stage-overlap txn=1 seqs=1-2 successor=2', 'x flash-s5 stage-overlap txn=2 seqs=5-7 successor=none']
        owner, faults = r.overlap_owners(texts)
        self.assertEqual(faults, [])
        chunks = r.regroup(self.flushes(), owner, {1: (0, 4), 2: (4, 8)})
        self.assertEqual([c['identity']['txn'] for c in chunks], [1, 2])
        self.assertEqual(sorted(x['seq'] for x in chunks[0]['rows']['n']), [1, 2])
        self.assertEqual(sorted(x['seq'] for x in chunks[1]['rows']['n']), [5, 6, 7])
        self.assertEqual([x['seq'] for x in chunks[0]['rows']['s']], [3])
        self.assertEqual((chunks[1]['identity']['chunk_start'], chunks[1]['identity']['chunk_end']), (4, 8))

    def test_a_flush_without_a_transaction_is_a_fault(self):
        one = self.flushes()[0]
        one['identity'].pop('txn')
        one.pop('txn')
        with self.assertRaisesRegex(ValueError, 'names no transaction'):
            r.regroup([one], {}, {})

    def test_without_overlap_lines_the_flush_is_the_chunk(self):
        chunks = r.regroup(self.flushes(), {}, {})
        self.assertEqual(sorted(x['seq'] for x in chunks[0]['rows']['n']), [1, 2, 5, 6])

    def test_a_sequence_named_by_two_chunks_is_a_fault(self):
        _, faults = r.overlap_owners(['flash-s5 stage-overlap txn=1 seqs=1-3 successor=2',
                                      'flash-s5 stage-overlap txn=2 seqs=3 successor=none'])
        self.assertEqual(faults, ['seq-3-named-by-two-chunks'])

    def test_slow_rank_per_stage(self):
        owner, _ = r.overlap_owners(['flash-s5 stage-overlap txn=1 seqs=1-2 successor=2',
                                     'flash-s5 stage-overlap txn=2 seqs=5-7 successor=none'])
        chunks = r.regroup(self.flushes(), owner, {})
        slow = r.slow_rank(chunks[1])
        # Stage 0 of txn 2: rank 0 spans 310-380 and 400-450 (120 ns), rank 1 310-350 and 400-500 (140 ns).
        self.assertEqual(slow['0']['per_rank_s'], {'0': 120e-9, '1': 140e-9})
        self.assertEqual(slow['0']['slow_rank'], '1')
        self.assertEqual((slow['1']['slow_rank'], slow['1']['slow_s']), ('3', 180e-9))
        summary = r.slow_summary([dict(slow_rank=slow), dict(slow_rank=r.slow_rank(chunks[0]))])
        self.assertEqual(summary['0']['slower_counts'], {'1': 2})
        self.assertEqual(summary['1']['chunks'], 2)

    def test_slow_rank_is_none_without_replies(self):
        chunk = dict(rows={'n': [n_row(1, 0, 100, {}, 120)]})
        self.assertEqual(r.slow_rank(chunk)['0'], dict(per_rank_s={'0': 0.0, '1': 0.0}, slow_rank=None, slow_s=None))


class RealLines(unittest.TestCase):
    """Fixture leaf `slow-stage-one` (harness/stage_overlap_fixture_leaves.json): stage 1 reads its ring every sixth tick, so
    stage 0 opens n+1 while stage 1 is on n; the lines are the engine's own emitters'. Box flushes name their transaction only in
    the identity, as here (the synthetic flushes above also carry it at top level)."""

    class Log:
        def __init__(self):
            self.lines = []

        def info(self, text, *args):
            self.lines.append(text % args if args else text)

    def run_stack(self):
        import e3_alpha_c0_box as box
        import engine_driver as ed
        import executor_hooks as eh
        import native_measurement as nm
        import stage_overlap as so
        import stage_progress as sp
        import test_e3_stage_overlap as t
        log = self.Log()
        states = {rank: (nm.ProgressTrace('worker', rank), collections.deque()) for rank in range(4)}
        with mock.patch.object(eh, '_worker_progress', lambda rank: states[rank]), mock.patch.object(eh, '_progress_logger', lambda: log), \
                mock.patch.object(ed, '_progress_logger', lambda: log), mock.patch.object(sp, '_progress_logger', lambda: log), \
                mock.patch.object(so, '_progress_logger', lambda: log):
            network = t.OverlapStack(direct=True, tokens=33)
            self.addCleanup(network.close_network)
            driver = network.driver
            driver.auto_arm = SimpleNamespace(manual_owned=False, phase='native', active=False)
            driver._admission_intent = dict(origin='auto-arm', phase='native', seq=1)
            driver.install_progress('burst')
            driver.progress.mode = 'burst'
            network.run_request(tick=t.Identity('test_slow_stage_one_matches_serial_and_runs_side_by_side').slow_tick(network))
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = Path(directory) / 'engine-stream.err'
        path.write_text(''.join('2026-11-01T11:00:%02d.%06dZ x %s\n' % (i // 1000000, i % 1000000, x) for i, x in enumerate(log.lines)))
        return box.progress, network, log.lines, path

    def test_interleaved_rows_return_to_their_chunk(self):
        progress, network, lines, path = self.run_stack()
        self.assertGreater(network.driver.overlap_counters['early_opens'], 0)
        records, errors = progress.read_lines([path])
        self.assertEqual(errors, [])
        flushes, problems = progress.assemble(records)
        self.assertEqual(problems, [])
        workers = {}
        for flush in flushes:
            if flush['side'] == 'worker':
                for row in flush['rows'].get('r', []):
                    workers[(row['seq'], flush['rank'])] = row
        engine = sorted((f for f in flushes if f['side'] == 'engine'), key=lambda f: f['line'])
        self.assertEqual(len(engine), network.controller.commits)
        # Without the regroup a flush holds its successor's early stage-0 rows (C0's per-flush count is then wrong).
        self.assertTrue(any(progress.chunk_breakdown(f, workers)['kinds'].get('open') != 2 for f in engine))
        owner, faults = r.overlap_owners(lines)
        self.assertEqual(faults, [])
        self.assertEqual(len([x for x in lines if 'stage-overlap' in x]), len(engine))
        chunks = r.regroup(engine, owner, {})
        self.assertEqual([c['identity'] for c in chunks], [f['identity'] for f in engine])
        self.assertEqual(sum(len(c['rows'].get('n', [])) for c in chunks), sum(len(f['rows'].get('n', [])) for f in engine))
        for index, chunk in enumerate(chunks):
            info = progress.chunk_breakdown(chunk, workers)
            with self.subTest(txn=chunk['identity']['txn']):
                self.assertEqual(info['foreign_replies'], 0)
                self.assertEqual((info['kinds']['open'], info['kinds']['release']), (2, 2))
                if index < len(chunks) - 1:
                    self.assertFalse(info['missing_worker_rows'], info['missing_worker_rows'])
                else:
                    self.assertTrue({row['kind'] for row in info['missing_worker_rows']} <= {'release'})


if __name__ == '__main__':
    unittest.main()
