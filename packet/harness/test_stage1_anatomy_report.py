#!/usr/bin/env python3
"""The stage-1 anatomy report on a synthetic record: hand-computed shares, the slower rank, torn and short lines, and
the lines the real recorder writes (from the CPU rig in test_stage1_handler_anatomy)."""
import contextlib
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'tools')]
import e3_native41_progress_box as box  # noqa: E402
import stage1_anatomy_report as report  # noqa: E402

MS = 1_000_000
STAMP = '2026-11-02T03:00:%02d.000000000Z '
T0 = 1_793_588_400   # 2026-11-02T03:00:00Z
SEAM = {name: index for index, name in enumerate(report.SEAMS)}


def seams(**cells):
    """name=(ns, excl, cpu, n) in ms (n plain) -> encoded cells."""
    return [[SEAM[k], int(v[0] * MS), int(v[1] * MS), int(v[2] * MS), v[3]] for k, v in sorted(cells.items())]


def main_row(seq, kind, enter_ms, wall_ms, cpu_ms, obs_ms, cells, ho_ms=0.1, dev_f=None, qb=None, qa=None):
    return [seq, kind, 1, 0, int(enter_ms * 1000), int((enter_ms + wall_ms) * 1000), int(wall_ms * MS),
            int(cpu_ms * MS), int(obs_ms * MS), None if dev_f is None else int(dev_f * MS), None, qb, qa,
            int(ho_ms * MS), cells]


def out_row(seq, thread, get_ms, get_cpu_ms, mq_ms):
    return [seq, thread, None, int(get_ms * MS), int(get_cpu_ms * MS), int(mq_ms * MS), None, 1000]


def handler_line(second, rank, line, main=(), out=(), part=0, parts=1, head=True):
    body = dict(schema=report.SCHEMA, rank=rank, line=line, base_ns=10 ** 12, part=part, parts=parts,
                rows=dict(m=list(main), a=list(out)))
    if head and part == 0:
        body['head'] = dict(identity=dict(rank=rank), calibration=dict(probe_ns=900), counters=dict(rows_dropped=0),
                            installed=['runner.execute_model'], missing=[], seams=list(report.SEAMS))
    return STAMP % second + 'INFO ' + report.HANDLER_LINE + json.dumps(body, separators=(',', ':'))


def engine_line(second, line, chunk_start, seqs, decode_live=8):
    body = dict(schema=box.PROGRESS_SCHEMA, side='engine', rank=None, line=line, base_ns=10 ** 12, part=0, parts=1,
                fields={k: list(v) for k, v in box.ROWS.items()}, chunk_start=chunk_start,
                chunk_end=chunk_start + 1024, decode_live=decode_live,
                rows=dict(s=[[seq, 'execute_model', 0, 1, 2] for seq in seqs]))
    return STAMP % second + 'INFO ' + box.PROGRESS_LINE + json.dumps(body, separators=(',', ':'))


def write_record(root, lines, arms=(('fused-live-0', 'fused', 'live'),)):
    with open(os.path.join(root, 'engine-current.log'), 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    for label, selection, cohort in arms:
        with open(os.path.join(root, 'arm-%s.json' % label), 'w') as handle:
            json.dump(dict(label=label, selection=selection, cohort=cohort, mode='burst', wall_start_s=T0,
                           wall_end_s=T0 + 59), handle)


# Rank 2 (the slower): 10 ms RPC, pp_recv 6 ms mostly blocked (0.5 ms CPU), forward 2 ms (device 2.2 ms, work queued
# after the launch, stream busy before it), execute 3.5 ms incl. the forward (excl 1.5 ms), observer 0.1 ms -> other
# 0.4 ms; handle_output 0.1 ms. Rank 3: the same with pp_recv 3 ms. sample_tokens on rank 2: 4 ms, bookkeeping 2 ms,
# handle_output 1 ms of which main-thread get_output 0.8 ms (0.05 ms CPU).
RANK2 = [main_row(201, 'e', 0, 10, 2, 0.1, seams(pp_recv=(6, 6, .5, 1), forward=(2, 2, 1, 1), execute=(3.5, 1.5, .3, 1)),
                  dev_f=2.2, qb=0, qa=0),
         main_row(202, 's', 11, 4, 1, 0.1, seams(sample_tokens=(3.9, 1.9, .8, 1), bookkeeping=(2, 2, .1, 1)),
                  ho_ms=1.0)]
RANK3 = [main_row(201, 'e', 0, 7, 2, 0.1, seams(pp_recv=(3, 3, .5, 1), forward=(2, 2, 1, 1), execute=(3.5, 1.5, .3, 1)),
                  dev_f=2.2, qb=0, qa=0),
         main_row(202, 's', 8, 4, 1, 0.1, seams(sample_tokens=(3.9, 1.9, .8, 1), bookkeeping=(2, 2, .1, 1)),
                  ho_ms=1.0)]
OUT2 = [out_row(202, 1, 0.8, 0.05, 0.1)]


class Synthetic(unittest.TestCase):
    def run_report(self, lines, **kw):
        with tempfile.TemporaryDirectory() as root:
            write_record(root, lines, **kw)
            return report.analyse(root, native=False)

    def lines(self):
        return [engine_line(1, 1, 0, [201, 202]),
                handler_line(2, 2, 0, RANK2, OUT2), handler_line(2, 3, 0, RANK3, [out_row(202, 1, .8, .05, .1)])]

    def test_rows_join_to_the_arm_and_band(self):
        got = self.run_report(self.lines())
        self.assertEqual(got['rows'], dict(main=4, output=2, failed=0, aborted=0, unassigned=0))
        self.assertEqual(sorted(got['groups']), ['fused-live-below-arena'])
        self.assertEqual(got['header']['2']['calibration'], dict(probe_ns=900))
        kinds = got['groups']['fused-live-below-arena']['per_rank_kind']
        self.assertEqual(kinds['2:execute_model']['wall_ms']['p50'], 10.0)
        self.assertEqual(kinds['2:execute_model']['seams_ms']['pp_recv']['excl']['p50'], 6.0)
        self.assertEqual(kinds['2:sample_tokens']['reply']['main_thread'], 1)
        self.assertAlmostEqual(kinds['2:execute_model']['other_ms']['p50'], 0.4, places=3)

    def test_slower_rank_and_carrier(self):
        got = self.run_report(self.lines())['groups']['fused-live-below-arena']['slower_tp_rank']['1']
        self.assertEqual(got['pairs'], 2)
        self.assertEqual(got['carrier_counts']['pp_recv'], 1)
        self.assertEqual(got['slower_rank_counts']['2'], 2 - got['slower_rank_counts']['3'])
        self.assertAlmostEqual(got['seam_difference_ms']['pp_recv']['p90'], 3.0)

    def test_decision_shares_by_hand(self):
        d = self.run_report(self.lines())['decision']['fused']['below-arena']
        # The slower rank per seq: 201 -> rank 2 (10.1 ms), 202 -> a tie (5 ms each), sorted stably -> either rank,
        # both identical. Totals: 10.1 + 5.0 = 15.1 ms.
        total = 15.1
        shares = d['category_shares']
        self.assertAlmostEqual(shares['pp_receive'], 6 / total, places=3)
        self.assertAlmostEqual(shares['gpu_cpu_sync'], (2 + .8) / total, places=3)
        self.assertAlmostEqual(shares['forward_launch'], 2 / total, places=3)
        self.assertAlmostEqual(shares['logits_sampling'], 1.9 / total, places=3)
        self.assertAlmostEqual(shares['output'], (.1 + .2) / total, places=3)
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=3)
        pp = d['outcomes']['pp_receive_or_gpu_cpu_sync']
        self.assertAlmostEqual(pp['share'], 8.8 / total, places=3)
        self.assertEqual(pp['class'], 'dominates')
        self.assertAlmostEqual(pp['blocked_share'], 1 - (.5 + .1 + .05) / (6 + 2 + .8), places=3)
        self.assertEqual(d['outcomes']['cpu_bound']['class'], 'minor')     # stream busy at launch: not counted
        gpu = d['outcomes']['waiting_on_decode_gpu']
        self.assertAlmostEqual(gpu['share'], 2 / total, places=3)            # 2 / 2.2 >= 0.8 and work queued after
        self.assertEqual(d['verdict'], 'dominates: pp_receive_or_gpu_cpu_sync')
        self.assertEqual(d['chunks'], 1)
        self.assertIn('no live chunks', self.run_report(self.lines())['decision']['fused']['beyond-arena']['verdict'])

    def test_no_dominant_outcome_names_the_largest(self):
        rows = [main_row(201, 'e', 0, 10, 2, 0.1, seams(pp_recv=(2, 2, .1, 1), execute=(3, 3, 3, 1),
                                                        forward=(4.9, 4.9, 1, 1)), dev_f=10, qb=1, qa=0)]
        d = self.run_report([engine_line(1, 1, 0, [201]), handler_line(2, 2, 0, rows), handler_line(2, 3, 0, rows)])
        d = d['decision']['fused']['below-arena']
        self.assertTrue(d['verdict'].startswith('no outcome reaches 50 %'), d['verdict'])
        self.assertIn('forward_launch', d['verdict'])
        self.assertEqual(d['outcomes']['waiting_on_decode_gpu']['class'], 'minor')   # 4.9 / 10 < 0.8

    def test_cpu_bound_counts_only_with_the_stream_idle_at_launch(self):
        rows = [main_row(201, 'e', 0, 10, 9, 0.1, seams(execute=(9, 9, 8.5, 1)), qb=1, qa=0)]
        d = self.run_report([engine_line(1, 1, 0, [201]), handler_line(2, 2, 0, rows), handler_line(2, 3, 0, rows)])
        cpu = d['decision']['fused']['below-arena']['outcomes']['cpu_bound']
        self.assertTrue(cpu['holds'])
        self.assertEqual(cpu['class'], 'dominates')

    def test_beyond_arena_band(self):
        lines = [engine_line(1, 1, 100_000, [201, 202])] + self.lines()[1:]
        got = self.run_report(lines)
        self.assertEqual(sorted(got['groups']), ['fused-live-beyond-arena'])
        self.assertEqual(got['decision']['fused']['beyond-arena']['chunks'], 1)

    def test_groups_split_fused_and_stepped(self):
        lines = [engine_line(1, 1, 0, [201, 202]), engine_line(40, 2, 0, [301])] + self.lines()[1:] + [
            handler_line(41, 2, 1, [main_row(301, 'e', 0, 5, 1, .1, [])])]
        with tempfile.TemporaryDirectory() as root:
            write_record(root, lines, arms=())
            for label, selection, start in (('fused-live-0', 'fused', T0), ('stepped-live-0', 'stepped', T0 + 30)):
                with open(os.path.join(root, 'arm-%s.json' % label), 'w') as handle:
                    json.dump(dict(label=label, selection=selection, cohort='live', wall_start_s=start,
                                   wall_end_s=start + 20), handle)
            got = report.analyse(root, native=False)
        self.assertEqual(sorted(got['groups']), ['fused-live-below-arena', 'stepped-live-below-arena'])
        self.assertEqual(sorted(got['decision']), ['fused', 'stepped'])

    def test_torn_line_is_counted_and_its_rows_dropped(self):
        lines = self.lines()
        good = lines[2]
        at = good.index('"rows"')
        lines[2] = good[:at] + '(Worker_TP1 pid=77) ' + good[at:]
        got = self.run_report(lines)
        self.assertEqual(got['torn_handler_lines'], 1)
        self.assertEqual(got['rows']['main'], 2)
        self.assertEqual(got['groups']['fused-live-below-arena']['slower_tp_rank']['1']['pairs'], 0)

    def test_short_flush_and_line_gap_are_counted(self):
        lines = self.lines()[:2] + [handler_line(2, 3, 0, RANK3, part=0, parts=2),
                                    handler_line(3, 2, 2, [main_row(203, 'o', 20, 1, 1, 0, [])])]
        got = self.run_report(lines)
        self.assertEqual(got['incomplete_flushes'], [dict(rank=3, line=0, parts=[0], of=2)])
        self.assertEqual(got['line_gaps'], [dict(rank=2, line=1)])
        self.assertEqual(got['rows']['unassigned'], 1)

    def test_cli_writes_a_fresh_file_only(self):
        with tempfile.TemporaryDirectory() as root:
            write_record(root, self.lines())
            out = os.path.join(root, 'out.json')
            with open(os.devnull, 'w') as null:
                with contextlib.redirect_stdout(null):
                    self.assertEqual(report.main([root, out, '--no-native']), 0)
                with open(out) as handle:
                    self.assertEqual(json.load(handle)['schema'], 'stage1-anatomy-report/1')
                with contextlib.redirect_stderr(null), self.assertRaises(SystemExit):
                    report.main([root, out, '--no-native'])


class RecorderLines(unittest.TestCase):
    """The lines the real recorder writes, from the CPU rig for ranks 2 and 3, read back by the report."""

    def test_rig_lines_read(self):
        import test_stage1_handler_anatomy as rigs
        texts, seqs = {}, []
        for rank in (2, 3):
            with contextlib.ExitStack() as stack:
                rig = rigs.Rig(stack, rank, costs=rigs.Fast)
                seqs = [s for _ in range(3) for s in rig.step()]
                rig.replies(6)
                rig.release()
                texts[rank] = rig.handler_lines()
        lines = [engine_line(1, 1, 4096, seqs)]
        for rank, items in texts.items():
            lines += [STAMP % 2 + 'INFO ' + item for item in items]
        with tempfile.TemporaryDirectory() as root:
            write_record(root, lines)
            got = report.analyse(root, native=False)
        self.assertEqual(got['torn_handler_lines'], 0)
        self.assertEqual(got['incomplete_flushes'], [])
        # Six RPCs per rank; each rank's first carries the install (calibration) and is flagged INSTALL: left out.
        self.assertEqual(got['rows']['main'], 10)
        self.assertEqual(got['rows']['unassigned'], 0)
        group = got['groups']['fused-live-below-arena']
        self.assertEqual(group['slower_tp_rank']['1']['pairs'], 5)
        self.assertIn('2:execute_model', group['per_rank_kind'])
        self.assertIn('forward', group['per_rank_kind']['2:execute_model']['seams_ms'])
        self.assertEqual(got['decision']['fused']['below-arena']['chunks'], 1)
        self.assertEqual(sorted(got['header']), ['2', '3'])


if __name__ == '__main__':
    unittest.main()
