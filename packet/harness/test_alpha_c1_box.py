#!/usr/bin/env python3
"""C1 grade (harness/e3_alpha_c1_box.py) over the real overlap composition's own lines. CPU only; no pass here is box evidence.

One connected run, built once: the sixteen declared arms in order, each one ``OverlapStack`` (the real overlap driver, staged
admission controller, worker bridges and emitters on the per-rank ring, harness/test_e3_stage_overlap.py) with the arm's
selection, stage 1 slow (so stage 0 opens n+1 early), and every line the run logs: engine and worker progress, stage-local,
stage-overlap, and the bridge's own ``flash-s5 memory`` receipts. The lines are stamped into the box's followed-log format and
graded by the real grade. The negative cases damage that run's lines or records the way the box can.

Fixture leaves (harness/stage_overlap_fixture_leaves.json): ``one-engine-numbering`` (each arm's stack continues the previous
one's command sequence and engine line numbers), ``decode-live-count``, ``fake-cuda-allocator``, ``selection-by-attribute``,
``arm-record-box-fields``, and C0's ``progress-log-lines``.
"""
import collections
import contextlib
import copy
import io
import json
import logging
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
import e3_alpha_c1_box as box  # noqa: E402
import engine_driver as ed  # noqa: E402
import executor_hooks as eh  # noqa: E402
import native_measurement as nm  # noqa: E402
import stage_overlap as so  # noqa: E402
import stage_progress as sp  # noqa: E402
import test_alpha_c0_queue as q0  # noqa: E402
import test_alpha_perf_queue as tap  # noqa: E402
import test_e3_stage_overlap as t  # noqa: E402
import worker_bridge as wb  # noqa: E402

progress, ap = box.progress, box.ap
WALL0 = 1793534190.0
TOKENS = 33
GIB = 2 ** 30
MIB = 2 ** 20
# Reserved bytes per selection (fake-cuda-allocator): the overlap arms hold 1 MiB more than serial, within the 2 MiB margin.
RESERVED = dict(serial=8 * GIB, local=8 * GIB, overlap=8 * GIB + MIB)
SPLICE = '(Worker_PP0_TP1_EP1 pid=102) '


class FakeCuda:
    """Leaf fake-cuda-allocator: torch.cuda's three host queries memory_receipt reads (is_initialized, memory_stats,
    mem_get_info). The receipt reads nothing else, so the bridge's emitter and line are the real ones."""

    def __init__(self, reserved):
        self.reserved = reserved

    def is_initialized(self):
        return True

    def memory_stats(self):
        return {'allocated_bytes.all.current': self.reserved - 64 * MIB, 'reserved_bytes.all.current': self.reserved,
                'inactive_split_bytes.all.current': 0, 'num_alloc_retries': 0, 'num_ooms': 0,
                'num_sync_all_streams': 0, 'num_device_alloc': 10, 'num_device_free': 2}

    def mem_get_info(self):
        return 24 * GIB - self.reserved, 24 * GIB


class Log(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def info(self, text, *args):
        self.lines.append(text % args if args else text)

    def emit(self, record):
        self.lines.append(record.getMessage())


class Docker:
    """Leaf progress-log-lines (C0's): the box's followed-log format; one line per millisecond, rank 3 on stderr."""

    def __init__(self):
        self.ms = 0
        self.lines = []

    def wall(self):
        return WALL0 + self.ms / 1000

    def log(self, text):
        self.ms += 1
        if text.startswith(progress.PROGRESS_LINE):
            body = json.loads(text[len(progress.PROGRESS_LINE):])
            who = 'engine' if body['side'] == 'engine' else body['rank']
        elif text.startswith(box.MEMORY_MARK):
            who = int(re.search(r'rank=(\d+)', text).group(1))
        else:
            who = 'engine'
        whole, frac = divmod(self.ms, 1000)
        stamp = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(int(WALL0) + whole)) + '.%09dZ' % (frac * 10 ** 6)
        prefix = ('(EngineCore_DP0 pid=101) INFO 11-01 12:00:00 [engine_driver.py:940]' if who == 'engine'
                  else '(Worker_TP%d pid=%d) INFO 11-01 12:00:00 [executor_hooks.py:1230]' % (who, 201 + who))
        self.lines.append(['err' if who == 3 else 'log', '%s %s %s\n' % (stamp, prefix, text)])


def run_arm(selection, cohort, carry):
    """One arm's stack and its lines. Returns (lines, before, after, commits)."""
    log = Log()
    logger = logging.getLogger(wb.__name__)
    level = logger.level
    logger.addHandler(log)
    logger.setLevel(logging.INFO)
    try:
        with mock.patch.object(eh, '_worker_progress', lambda rank: carry['states'][rank]), \
                mock.patch.object(eh, '_progress_logger', lambda: log), mock.patch.object(ed, '_progress_logger', lambda: log), \
                mock.patch.object(sp, '_progress_logger', lambda: log), mock.patch.object(so, '_progress_logger', lambda: log):
            network = t.OverlapStack(direct=True, tokens=TOKENS, overlap=selection == 'overlap')
            driver = network.driver
            driver.stage_selected = selection != 'serial'          # leaf selection-by-attribute
            network.transport._next = carry['next']                # leaf one-engine-numbering
            for rank in range(4):
                bridge = network.workers[rank]._flash_s5_bridge
                bridge._memory_receipts = True                     # FLASH_S5_AUTO_ARM=1 at bridge construction
                bridge._memory_cuda = FakeCuda(RESERVED[selection])
            calls = [0]

            def decode_live():                                     # leaf decode-live-count
                calls[0] += 1
                return 8 if cohort == 'live' or calls[0] == 1 else 0
            driver._progress_decode_live = decode_live
            driver.auto_arm = SimpleNamespace(manual_owned=False, phase='native', active=False)
            driver._admission_intent = dict(origin='auto-arm', phase='native', seq=1)
            driver.install_progress('burst')
            driver.progress.mode = 'burst'
            driver.progress.trace.lines_out = carry['engine_lines']
            before = driver.stage_status()
            tick = t.Identity('test_slow_stage_one_matches_serial_and_runs_side_by_side').slow_tick(network)
            network.run_request(tick=tick)
            after = driver.stage_status()
            carry['next'] = network.transport._next
            carry['engine_lines'] = driver.progress.trace.lines_out
            commits = network.controller.commits
            assert network.worker_errors == [], network.worker_errors
            network.close_network()
    finally:
        logger.removeHandler(log)
        logger.setLevel(level)
    return log.lines, before, after, commits


def build_run():
    docker, arms, spans = Docker(), [], {}
    carry = dict(states={rank: (nm.ProgressTrace('worker', rank), collections.deque()) for rank in range(4)}, next=1, engine_lines=0)
    for selection, cohort, index in box.ARM_ORDER:
        name = box.label(selection, cohort, index)
        docker.ms += 5000                                          # beyond the join's 2 s log lag between arms
        start, first = docker.wall(), len(docker.lines)
        lines, before, after, commits = run_arm(selection, cohort, carry)
        for text in lines:
            docker.log(text)
        docker.ms += 10
        end = docker.wall()
        spans[name] = (first, len(docker.lines))
        view = lambda status: box.stage_view({'native_stage_local': status})  # noqa: E731
        streams = [40.0, 41.0, 42.0, 42.0, 43.0, 43.0, 44.0, 45.0]
        client = {'prefill_tokens_per_s': 900.0, 'all_open_share_of_span': 1.0 if cohort == 'live' else 0.1}
        if cohort == 'live':
            client['decode_window'] = {'aggregate_tokens_per_s': 340.0, 'stream_rates_tokens_per_s': streams, 'gap_ms_p99': 60.0,
                                       'gap_ms_max': 100.0}
        # Leaf arm-record-box-fields: the client-side fields of the arm record; the selection, status deltas, spans and chunk
        # counts are the run's own.
        arms.append(dict(label=name, selection=selection, mode='burst', cohort=cohort, index=index, band=ap.band_of_index(index),
                         complete=True, reasons=[], wall_start_s=start, wall_end_s=end,
                         progress_delta={'chunks': commits, 'commands': commits * 16},
                         stage_delta=box.stage_delta(view(after), view(before)),
                         selection_switch=dict(local=selection != 'serial', overlap=selection == 'overlap', route='full'),
                         outputs=q0.outputs(selection, index), fresh={'prompt_tokens_observed': TOKENS, 'ttft_s': 1.0},
                         decode_only={'aggregate_tokens_per_s': 400.0, 'stream_rates_tokens_per_s': [50.0] * 8},
                         client_basis=client, append={'ttft_s': 2.0}, metrics={}))
    return docker.lines, arms, spans


_RUN = []


def real_run():
    if not _RUN:
        _RUN.append(build_run())
    lines, arms, spans = _RUN[0]
    return copy.deepcopy(lines), copy.deepcopy(arms), spans


class Base(unittest.TestCase):
    def setUp(self):
        self.lines, self.arms, self.spans = real_run()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def grade(self, lines=None, arms=None):
        q = Path(tempfile.mkdtemp(dir=self.tmp.name))
        tap.write_streams(q, self.lines if lines is None else lines)
        arms = self.arms if arms is None else arms
        return box.grade(arms, {a['label']: tap.PASS for a in arms}, tap.CHECKS,
                         [str(q / 'engine-stream.log'), str(q / 'engine-stream.err')], {a['label']: tap.ALIVE for a in arms})

    def inside(self, name, mark):
        lo, hi = self.spans[name]
        return [i for i in range(lo, hi) if mark in self.lines[i][1]]

    def arm(self, name):
        return next(a for a in self.arms if a['label'] == name)

    @staticmethod
    def splice(line, after):
        stream, text = line
        at = text.index(after) + len(after)
        return [stream, text[:at] + SPLICE + text[at:]]


class Connected(Base):
    def test_sixteen_real_arms_pass(self):
        result = self.grade()
        self.assertEqual((result['verdict'], result['failures'], result['missing'], result['classification']),
                         ('PASS', [], [], None), result['missing'])
        self.assertEqual(result['required_arms'], [box.label(*x) for x in box.ARM_ORDER])
        self.assertEqual(len(result['required_arms']), 16)
        self.assertEqual([box.label(*x) for x in box.BAND1_ORDER],
                         ['serial-solo-0', 'local-solo-0', 'overlap-solo-0', 'overlap-live-0', 'local-live-0', 'serial-live-0',
                          'serial-live-1', 'local-live-1', 'overlap-live-1', 'overlap-solo-1', 'local-solo-1', 'serial-solo-1'])
        self.assertEqual(box.BAND1_ORDER[6:], tuple((s, c, 1) for s, c, _ in reversed(box.BAND1_ORDER[:6])))  # mirrored rounds
        self.assertEqual(result['schema'], 'alpha-c1-grade/1')
        for name, v in result['validity'].items():
            e, f = v['engagement'], v['in_flight']
            with self.subTest(arm=name):
                self.assertEqual((v['verdict'], e['engaged']), ('PASS', True))
                self.assertEqual([len(f[s]['breaches']) for s in ('0', '1')], [0, 0])
                self.assertGreater(min(f[s]['commands'] for s in ('0', '1')), 0)
                self.assertEqual(v['oom_rises'], [])
                memory = result['table'][name]['memory']
                self.assertEqual((memory['torn_receipts'], memory['lost_receipts'], memory['established']), (0, [], True))
                self.assertEqual(len(memory['chunks']), e['owned_chunks'])
                if name.startswith('overlap'):
                    self.assertGreater(e['overlap']['early_opens'], 0)
                    self.assertEqual(e['overlap_lines'], e['owned_chunks'])
                    self.assertEqual((e['regrouped_chunks_checked'], e['regrouped_chunks_uneven']), (e['owned_chunks'], []))
                    self.assertGreater(v['regroup']['moved_rows'], 0)
                else:
                    self.assertEqual((e['overlap_lines'], set(e['overlap'].values())), (0, {0}))
        pair = result['memory_vs_serial']['overlap-live-3']
        self.assertEqual((pair['read'], pair['serial'], pair['zero_ooms'], pair['min_free_within_margin']),
                         ('MET', 'serial-live-3', True, True))
        self.assertEqual((pair['transient']['max_reserved_delta_mib'], pair['transient']['min_free_delta_mib']), (1.0, -1.0))
        self.assertEqual(pair['transient']['chunks'], 8)
        self.assertEqual(result['memory_vs_serial']['local-live-1']['transient']['max_reserved_delta_mib'], 0.0)
        # Local solo arms (lead, 2 Nov) have their matched serial arm and their speed group against serial.
        self.assertEqual([result['memory_vs_serial'][x]['serial'] for x in ('local-solo-0', 'local-solo-1')],
                         ['serial-solo-0', 'serial-solo-1'])
        self.assertEqual(result['memory_vs_serial']['local-solo-1']['read'], 'MET')
        self.assertIn('local-solo-1', result['summary']['over_serial'])
        self.assertEqual(sorted(result['memory_vs_serial']),
                         sorted(box.label(*x) for x in box.ARM_ORDER if x[0] != 'serial'))
        slow = result['table']['overlap-live-0']['slow_rank']
        for stage in ('0', '1'):
            self.assertIsNotNone(slow['all'][stage]['median_slow_s'])
            self.assertGreaterEqual(slow['all'][stage]['median_slow_s'], slow['all'][stage]['median_pooled_s'])
            self.assertEqual(slow['all'][stage]['chunks'], 8)
        self.assertEqual(sum(b['0']['chunks'] for b in slow['bands'].values()), 8)
        self.assertEqual(result['numerics']['verdict'], 'IDENTICAL')
        self.assertEqual(sorted(result['numerics']['parts']), ['local-band1', 'overlap-band1', 'overlap-band2'])
        groups = result['summary']['groups']
        self.assertIn('overlap-live-2', groups)
        self.assertIn('overlap-live-1', result['summary']['over_serial'])
        self.assertEqual((result['memory_is_gate'], result['oom_is_gate'], result['speed_is_gate']), (False, True, False))
        text = box.report(result)
        for heading in ('# q18a-alpha-c1 report', '## Mechanism and numerics', '## Performance',
                        '### Overlap engagement and one command in flight per stage', '### Device memory against the matched serial arm',
                        '### Slower TP rank per stage', '## Release readiness'):
            self.assertEqual(text.count(heading + '\n'), 1, heading)
        self.assertNotIn('local against the matched serial arm', text)

    def test_without_the_regroup_an_overlap_arm_is_not_engaged(self):
        with mock.patch.object(box, 'regrouping', lambda original: original):
            result = self.grade()
        self.assertEqual(result['verdict'], 'INC')
        reasons = result['validity']['overlap-live-0']['reasons']
        self.assertIn('regrouped-chunk-without-two-opens-and-two-releases', reasons)
        self.assertEqual(result['validity']['serial-live-0']['verdict'], 'PASS')


class Engagement(Base):
    def test_no_early_open_or_no_successor_is_inc(self):
        for field, reason in (('overlap_early_opens', 'overlap-stage-zero-never-opened-a-successor-early'),
                              ('overlap_successor_frames', 'overlap-ran-no-second-chunk')):
            with self.subTest(field=field):
                arms = copy.deepcopy(self.arms)
                self.arm_of(arms, 'overlap-solo-2')['stage_delta'][field] = 0
                result = self.grade(arms=arms)
                v = result['validity']['overlap-solo-2']
                self.assertEqual((result['verdict'], v['verdict']), ('INC', 'INC'))
                self.assertIn(reason, v['reasons'])

    @staticmethod
    def arm_of(arms, name):
        return next(a for a in arms if a['label'] == name)

    def test_a_local_or_serial_arm_moving_an_overlap_counter_or_logging_its_line_is_inc(self):
        arms = copy.deepcopy(self.arms)
        self.arm_of(arms, 'local-live-0')['stage_delta']['overlap_successor_frames'] = 1
        lines = list(self.lines)
        source = self.inside('overlap-live-0', box.OVERLAP_MARK)[0]
        target = self.inside('serial-live-0', progress.PROGRESS_LINE)[-1]
        stamp = lines[target][1].split(' ', 1)[0]
        lines.insert(target, [lines[source][0], stamp + ' ' + lines[source][1].split(' ', 1)[1]])
        result = self.grade(lines=lines, arms=arms)
        self.assertIn('local-arm-moved-an-overlap-counter', result['validity']['local-live-0']['reasons'])
        self.assertIn('serial-arm-logged-an-overlap-line', result['validity']['serial-live-0']['reasons'])
        self.assertEqual(result['verdict'], 'INC')

    def test_a_missing_overlap_line_is_inc(self):
        lines = list(self.lines)
        del lines[self.inside('overlap-live-1', box.OVERLAP_MARK)[3]]
        result = self.grade(lines=lines)
        v = result['validity']['overlap-live-1']
        self.assertEqual(v['verdict'], 'INC')
        self.assertIn('overlap-lines-differ-from-the-auto-owned-chunks', v['reasons'])
        self.assertIn('regrouped-chunk-without-two-opens-and-two-releases', v['reasons'])

    def test_a_torn_overlap_line_drops_its_chunk_and_the_older_one_within_the_limits(self):
        lines = list(self.lines)
        i = self.inside('overlap-live-1', box.OVERLAP_MARK)[3]
        lines[i] = self.splice(lines[i], 'seqs=')
        result = self.grade(lines=lines)
        v = result['validity']['overlap-live-1']
        self.assertEqual((result['verdict'], v['verdict']), ('PASS', 'PASS'), v['reasons'])
        self.assertEqual((v['tears']['torn_overlap_lines'], v['tears']['lost_records'], len(v['tears']['dropped_chunk_starts'])),
                         (1, 1, 2))
        self.assertEqual(v['engagement']['lost_overlap_lines'], 1)
        # A second torn line in the same arm is above the declared limits.
        j = self.inside('overlap-live-1', box.OVERLAP_MARK)[5]
        lines[j] = self.splice(lines[j], 'seqs=')
        v = self.grade(lines=lines)['validity']['overlap-live-1']
        self.assertEqual(v['verdict'], 'INC')
        self.assertIn('progress:lost-records-above-the-declared-limit:2>1', v['reasons'])

    def test_a_lost_engine_flush_on_an_overlap_arm_leaves_rows_without_their_chunk(self):
        lines = list(self.lines)
        del lines[self.inside('overlap-solo-0', '"side":"engine"')[4]]
        v = self.grade(lines=lines)['validity']['overlap-solo-0']
        self.assertEqual(v['verdict'], 'INC')
        self.assertIn('progress:overlap-rows-of-a-chunk-without-its-flush', v['reasons'])
        self.assertGreater(v['regroup']['orphan_rows'], 0)

    def test_a_sequence_named_by_two_chunks_is_inc(self):
        lines = list(self.lines)
        i, j = self.inside('overlap-live-0', box.OVERLAP_MARK)[1:3]
        first = re.search(r'seqs=(\d+)', lines[i][1]).group(1)
        lines[j][1] = re.sub(r'seqs=', 'seqs=%s,' % first, lines[j][1], count=1)
        v = self.grade(lines=lines)['validity']['overlap-live-0']
        self.assertEqual(v['verdict'], 'INC')
        self.assertTrue(any(x.startswith('progress:overlap-regroup:seq-') for x in v['reasons']), v['reasons'])


class InFlight(Base):
    def test_a_stage_command_submitted_before_the_previous_one_applied_fails_engine(self):
        lines = list(self.lines)
        for i in self.inside('overlap-live-0', '"side":"engine"'):
            stream, text = lines[i]
            head, _, body = text.partition(progress.PROGRESS_LINE)
            data = json.loads(body)
            rows = sorted((r for r in data['rows'].get('n', []) if r[1] == 'segment' and r[4] == 0), key=lambda r: r[7])
            if len(rows) >= 2 and type(rows[0][13]) is int:
                rows[1][7] = rows[0][13] - 1
                lines[i] = [stream, head + progress.PROGRESS_LINE + json.dumps(data, separators=(',', ':'), sort_keys=True) + '\n']
                break
        result = self.grade(lines=lines)
        v = result['validity']['overlap-live-0']
        self.assertEqual((result['verdict'], result['classification'], v['verdict'], v['classification']),
                         ('FAIL', ['ENGINE'], 'FAIL', 'ENGINE'))
        self.assertIn('two-commands-in-flight-on-stage-0', v['reasons'])
        self.assertEqual(len(v['in_flight']['0']['breaches']), 1)

    def test_the_rule_on_its_own(self):
        rows = [(0, 0, 10, 1), (1, 5, 30, 2), (0, 10, None, 3), (0, 20, 25, 4), (1, 29, 40, 5)]
        out = box.in_flight(rows)
        self.assertEqual((out['0']['commands'], out['0']['unbounded'], out['0']['breaches']), (3, 1, []))
        self.assertEqual(out['1']['breaches'], [dict(seq=5, previous_seq=2, early_s=1e-9)])


class Memory(Base):
    def memory_lines(self, name, rank=None):
        return [i for i in self.inside(name, box.MEMORY_MARK) if rank is None or 'rank=%d ' % rank in self.lines[i][1]]

    @staticmethod
    def with_field(line, field, value):
        return [line[0], re.sub(r'\b%s=-?\d+' % field, '%s=%d' % (field, value), line[1], count=1)]

    def test_a_torn_receipt_holding_the_minimum_makes_the_read_inc(self):
        lines = list(self.lines)
        i = self.memory_lines('overlap-live-3', 2)[5]
        low = (24 * GIB - RESERVED['overlap']) - 512 * MIB
        lines[i] = self.with_field(lines[i], 'free', low)
        # Whole, it is the minimum and the read is NOT_MET.
        pair = self.grade(lines=lines)['memory_vs_serial']['overlap-live-3']
        self.assertEqual((pair['read'], pair['min_free_within_margin']), ('NOT_MET', False))
        # Torn after the number (C0's reading of `free=` would still take it): not read, and the read is INC, never MET.
        lines[i] = self.splice(lines[i], 'total=')
        result = self.grade(lines=lines)
        pair = result['memory_vs_serial']['overlap-live-3']
        self.assertEqual((pair['read'], pair['reason']), ('INC', 'receipt-torn-or-lost'))
        self.assertEqual(pair['candidate']['min_free_mib'], round((24 * GIB - RESERVED['overlap']) / MIB, 1))
        self.assertEqual(result['table']['overlap-live-3']['memory']['torn_receipts'], 1)
        self.assertEqual(result['verdict'], 'PASS')        # a reading, never a gate

    def test_a_receipt_cut_before_device_frees_is_counted_never_skipped(self):
        # The lead's condition on whole-receipts-only: a line that simply ends before ` device_frees=` (no foreign bytes) is a
        # torn receipt, so the read is INC; its fields up to free= are not read.
        lines = list(self.lines)
        i = self.memory_lines('overlap-live-1', 3)[4]
        stream, text = lines[i]
        lines[i] = [stream, text[:text.index(' device_frees=')]]
        self.assertIn(box.MEMORY_MARK, lines[i][1])
        result = self.grade(lines=lines)
        memory = result['table']['overlap-live-1']['memory']
        self.assertEqual((memory['torn_receipts'], memory['established']), (1, False))
        pair = result['memory_vs_serial']['overlap-live-1']
        self.assertEqual((pair['read'], pair['reason']), ('INC', 'receipt-torn-or-lost'))
        self.assertIn('max allocated MiB', box.report(result))

    def test_a_torn_extra_receipt_beside_a_whole_one_still_makes_the_read_inc(self):
        lines = list(self.lines)
        i = self.memory_lines('overlap-solo-1', 0)[3]
        lines.insert(i + 1, self.splice(lines[i], 'reserved='))    # a second, torn copy: nothing is lost, one line is torn
        result = self.grade(lines=lines)
        memory = result['table']['overlap-solo-1']['memory']
        self.assertEqual((memory['torn_receipts'], memory['lost_receipts']), (1, []))
        self.assertEqual(result['memory_vs_serial']['overlap-solo-1']['read'], 'INC')

    def test_a_lost_receipt_makes_the_read_inc_and_the_serial_base_too(self):
        lines = list(self.lines)
        del lines[self.memory_lines('overlap-solo-0', 1)[2]]
        del lines[self.memory_lines('serial-live-1', 0)[0]]
        result = self.grade(lines=lines)
        cand = result['memory_vs_serial']['overlap-solo-0']
        self.assertEqual((cand['read'], cand['reason']), ('INC', 'receipt-torn-or-lost'))
        self.assertEqual(len(result['table']['overlap-solo-0']['memory']['lost_receipts']), 1)
        for name in ('local-live-1', 'overlap-live-1'):
            self.assertEqual(result['memory_vs_serial'][name]['read'], 'INC', name)
            self.assertEqual(result['memory_vs_serial'][name]['reason'], 'serial-arm-receipt-torn-or-lost')
        self.assertEqual(result['memory_vs_serial']['overlap-live-0']['read'], 'MET')

    def test_an_oom_in_an_intact_or_a_torn_receipt_fails_the_arm_as_engine(self):
        for torn in (False, True):
            with self.subTest(torn=torn):
                lines = list(self.lines)
                i = self.memory_lines('overlap-live-0', 3)[4]
                lines[i] = self.with_field(lines[i], 'ooms', 1)
                if torn:
                    lines[i] = self.splice(lines[i], 'reserved=')
                result = self.grade(lines=lines)
                v = result['validity']['overlap-live-0']
                self.assertEqual((result['verdict'], result['classification'], v['verdict'], v['classification']),
                                 ('FAIL', ['ENGINE'], 'FAIL', 'ENGINE'))
                self.assertIn('oom', v['reasons'])
                self.assertEqual([(r['rank'], r['ooms'], r['torn']) for r in v['oom_rises']], [(3, 1, torn)])
                # The later receipts still read 0: one rise, one arm.
                self.assertEqual(sum(1 for x in result['validity'].values() if x['oom_rises']), 1)

    def test_an_oom_outside_every_arm_fails_the_run(self):
        lines = list(self.lines)
        i = self.memory_lines('overlap-live-0', 3)[0]
        lo, _ = self.spans['overlap-live-0']
        stamp = lines[lo - 1][1].split(' ', 1)[0] if lo else None
        moved = self.with_field(lines[i], 'ooms', 1)
        second = int(stamp[17:19])
        moved[1] = stamp[:17] + '%02d' % (second + 3) + stamp[19:] + ' ' + moved[1].split(' ', 1)[1]
        lines.insert(lo, moved)
        result = self.grade(lines=lines)
        self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))
        self.assertIn('oom-outside-every-arm-span', result['failures'])

    def test_the_transient_past_the_arena_is_reported_apart(self):
        def memory(free, reserved, retries=False):
            chunks = [dict(chunk_start=a, chunk_end=b, txn=i, receipts=8, peak_reserved_mib=reserved + (a >= box.ARENA_ROWS) * 300,
                           min_free_mib=free - (a >= box.ARENA_ROWS) * 300, retries_moved=retries and a == 98176)
                      for i, (a, b) in enumerate(((0, 97152), (97152, 98176), (98176, 131093)))]
            worst = dict(peak_allocated_mib=1.0, peak_reserved_mib=reserved + 300, min_free_mib=free - 300, alloc_retries=0, ooms=0,
                         receipts=24)
            return dict(worst=worst, chunks=chunks, established=True)
        table = {'serial-live-3': dict(memory=memory(1000.0, 8000.0)), 'overlap-live-3': dict(memory=memory(999.0, 8001.0, True))}
        row = box.memory_compare(table)['overlap-live-3']
        self.assertEqual(row['read'], 'MET')
        self.assertEqual(row['transient'], dict(chunks=3, max_reserved_delta_mib=1.0, min_free_delta_mib=-1.0, chunks_with_retries=1))
        self.assertEqual(row['past_arena'], dict(chunks=2, max_reserved_delta_mib=1.0, min_free_delta_mib=-1.0, chunks_with_retries=1))
        table['overlap-live-3']['memory']['chunks'][2]['min_free_mib'] -= 5
        self.assertEqual(box.memory_compare(table)['overlap-live-3']['past_arena']['min_free_delta_mib'], -6.0)

    def test_fragments_of_a_torn_line(self):
        text = ('x flash-s5 memory point=open rank=1 txn_seq=3 chunk=0-4 allocated=1 reser' + SPLICE
                + 'flash-s5 memory point=open rank=2 txn_seq=3 chunk=0-4 allocated=1 reserved=2 inactive_split=0 alloc_retries=0 ooms=4')
        self.assertEqual(box._fragments(text), [(1, None), (2, 4)])
        self.assertEqual(box._fragments('flash-s5 memory point=open ra' + SPLICE + 'ooms=2 free=1'), [(None, 2)])


class Signatures(Base):
    def test_each_overlap_contract_error_fails_that_arm_as_engine(self):
        for name, text in box.OVERLAP_SIGNATURES:
            with self.subTest(signature=name):
                lines = list(self.lines)
                i = self.inside('overlap-solo-1', progress.PROGRESS_LINE)[4]
                stamp = lines[i][1].split(' ', 1)[0]
                lines.insert(i, ['log', '%s (EngineCore_DP0 pid=101) ERROR 11-01 12:00:00 [engine_driver.py:1] DriverError: %s\n'
                                 % (stamp, text)])
                v = self.grade(lines=lines)['validity']['overlap-solo-1']
                self.assertEqual((v['verdict'], v['classification']), ('FAIL', 'ENGINE'))
                self.assertIn(name, v['signatures'])


class Bands(Base):
    def test_numerics_compare_within_a_band(self):
        arms = copy.deepcopy(self.arms)
        for a in arms:
            if a['band'] == 2:
                a['outputs']['fresh'] = [99]
        self.assertEqual(self.grade(arms=arms)['numerics']['verdict'], 'IDENTICAL')
        next(a for a in arms if a['label'] == 'overlap-solo-2')['outputs']['fresh'] = [98]
        result = self.grade(arms=arms)
        self.assertEqual((result['verdict'], result['classification'], result['numerics']['differs']),
                         ('FAIL', ['ENGINE'], ['overlap-band2:fresh']))

    def test_a_declared_arm_not_run_is_inc_never_an_engine_result(self):
        arms = [a for a in self.arms if a['label'] != 'serial-live-3']
        result = self.grade(arms=arms)
        self.assertEqual((result['verdict'], result['failures']), ('INC', []))
        self.assertEqual(result['validity']['serial-live-3']['reasons'], ['no-arm-record'])
        self.assertEqual(result['memory_vs_serial']['overlap-live-3']['read'], 'INC')


class Selection(unittest.TestCase):
    class Client:
        def __init__(self, overlap_reply=None):
            self.calls, self.overlap_reply = [], overlap_reply

        def pause(self):
            self.calls.append(('pause', {}))
            return {'route': 'full', 'route_generation': 1}

        def resume(self):
            self.calls.append(('resume', {}))
            return {'resumed': True}

        def call(self, action, **fields):
            self.calls.append((action, fields))
            if action == 'native_progress':
                return {'mode': fields['mode'], 'installed': 'burst', 'route': 'full'}
            if action == 'native_stage_local':
                overlap = fields['overlap'] if self.overlap_reply is None else self.overlap_reply
                return {'local': fields['local'], 'overlap': overlap, 'previous_overlap': True, 'route': 'full'}
            return {'released': True}

    def test_each_selection_sends_both_fields_and_proves_both(self):
        for selection, local, overlap in (('serial', False, False), ('local', True, False), ('overlap', True, True)):
            with self.subTest(selection=selection):
                client, record = self.Client(), {}
                box.release_selected(selection)(client, record, 'burst')
                self.assertIn(('native_stage_local', dict(local=local, overlap=overlap)), client.calls)
                self.assertEqual((record['selection_switch']['local'], record['selection_switch']['overlap']), (local, overlap))

    def test_a_reply_with_the_other_overlap_is_refused(self):
        client, record = self.Client(overlap_reply=False), {}
        with self.assertRaisesRegex(ValueError, 'native-stage-overlap-selection-unproved'):
            box.release_selected('overlap')(client, record, 'burst')
        self.assertEqual(client.calls[-1][0], 'resume')

    def test_the_status_needs_the_overlap_block_and_the_receipts_check_it(self):
        status = {'native_stage_local': dict(selected=True, counters=dict.fromkeys(box.c0.STAGE_COUNTERS, 0))}
        with self.assertRaisesRegex(ValueError, 'native-stage-overlap-not-installed'):
            box.stage_view(status)
        status['native_stage_local']['overlap'] = dict(selected=False, counters=dict.fromkeys(box.OVERLAP_COUNTERS, 0))
        view = box.stage_view(status)
        record = {'control_history': list(box.c0.CONTROL_SEQUENCE), 'progress': [
            dict(label='before-fresh', native_stage_local=view), dict(label='after-fresh', native_stage_local=view)]}
        with mock.patch.object(box.c0, 'stage_delta', box.stage_delta):
            self.assertIn('stage-overlap-selection-not-this-arm', box.selection_reasons(record, 'overlap'))
            self.assertNotIn('stage-overlap-selection-not-this-arm', box.selection_reasons(record, 'local'))

    def test_the_arm_cli_takes_the_three_selections_and_four_indexes(self):
        with tempfile.NamedTemporaryFile() as out:
            common = ['arm', '--url', 'http://127.0.0.1:1', '--model', 'm', '--run-id', 'r', '--epoch', '1', '--cohort', 'live',
                      '--workload', 'w.json', '--out', out.name]
            self.assertEqual(box.main(common + ['--mode', 'overlap', '--index', '3']), 2)      # parsed; the existing out refuses
            for bad in (['--mode', 'fast', '--index', '0'], ['--mode', 'local', '--index', '4']):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    box.main(common + bad)

    def test_c0_names_are_restored(self):
        before = (box.c0.release_selected, box.c0.stage_view, box.c0.engagement, box.c0.ARM_ORDER, box.progress.assemble)
        with box._bound(box.c0, box.arm_names('overlap')):
            self.assertIs(box.c0.stage_view, box.stage_view)
        self.assertFalse(hasattr(box.c0, 'print'))
        self.assertEqual(before, (box.c0.release_selected, box.c0.stage_view, box.c0.engagement, box.c0.ARM_ORDER,
                                  box.progress.assemble))


if __name__ == '__main__':
    unittest.main()
