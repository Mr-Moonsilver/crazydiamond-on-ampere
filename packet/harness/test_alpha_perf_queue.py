"""Alpha-perf queue: band arms over the real control plane, the grade over real emitters, the arm loop in bash. CPU only.

Classes:

* Arm: the derived draft's exact arm command line, both bands, through two NativeRuntimeClients over loopback HTTP into
  the real endpoint, session, control client, driver, AutoArmPolicy and NativeProgress of a native38 startup-bound
  stack. Band 2 sends the declared 32,768 standing budget and band 1 held-f's 16,384; the wrong manifest for a round
  is refused before any traffic. As in held-f's Arm, nothing links the HTTP prompt to the scheduler, so no native chunk
  runs here (the chunk receipts on a connected Core are e3_native41_progress_probe.py in the engine self-run).
* Grade: sixteen arms of lines written by the real nm.ProgressTrace emitters (held-f's Emit) plus memory and retry
  lines in the engine's format, chunks across the 97,152-row boundary: the table per arm and band, the per-arm
  validity PASS/FAIL (crash-free, 0 disagree), the budget skip, each defect.
* ArmLoop: the derived body under real bash with the draft's own step, record, host_crash_detected, engine_watched and
  native41_budget, the real engine-death watch and the real grade, over stub arm and check clients: env reaches the
  client through engine_watched, the band picks the manifest, the budget guard, a dead engine stops the arm.
* Commands / QueueArgparse: every python3 call of the derived texts parses with its own module's real parser.
* Derivation / Workload / Window: derived texts on disk and parse; EXTRA_ENV carries the three flags; the band 2
  manifest and its length declaration; the phase budget and the slot at real wall clock.

Fixture leaves: alpha_perf_fixture_leaves.json, beside the inherited files it names.
"""
import argparse
import contextlib
import difflib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
import e3_alpha_perf_box as ap  # noqa: E402
import alpha_perf_gate as gate  # noqa: E402
import alpha_perf_queue as queue  # noqa: E402
import e3_native39_w2_length as length  # noqa: E402
import native43_le_fix_queue_deltas as le_fix  # noqa: E402
import test_native41_box_queue as held  # noqa: E402
import test_native41_progress_queue as hf  # noqa: E402

progress = ap.progress
SLOT_RECEIPTS = []
ARGPARSE_RECEIPTS = []
DRAFT = 'harness/runs/q18a_alpha_perf.draft.sh'
SETTINGS = 'harness/runs/q18a_alpha_perf_settings.sh'
DECLARE = 'harness/runs/q18a_alpha_perf_declare.sh'
LABELS = [progress.label(*x) for x in ap.ARM_ORDER]
BAND2_LABELS = [progress.label(*x) for x in ap.BAND2_REQUIRED + ap.BAND2_OPTIONAL]
DEATH_LINE = '2026-11-01T22:00:00.000000000Z (Worker_TP0 pid=201) EngineCore encountered a fatal error\n'
GPU = {'band1': (22000 * 2 ** 20, 23000 * 2 ** 20), 'band2': (22500 * 2 ** 20, 23300 * 2 ** 20)}


def band_cuts(band):
    # Band 2 chunks run across every history band past the arena (test scale: a few wide chunks, same row numbers).
    return (0, 1000, 2000, 3000) if band == 1 else (0, 97152, 98176, 114688, 131072, 131093)


def body_commands(text):
    """Every python3 command of the derived body, continuations joined, redirections dropped."""
    body = text[text.index(queue.BODY.splitlines()[0]):]
    body = body[:body.index('record alpha-perf')]
    joined = re.sub(r'\\\n\s*', ' ', body)
    rows = []
    for line in joined.splitlines():
        match = re.search(r'python3 "\$H/([a-z0-9_]+)\.py"(.*)$', line)
        if match:
            rows.append((match.group(1), re.split(r'\s+>\s*"', match.group(2))[0].strip()))
    return rows


QUEUE_VALUES = dict(hf.QUEUE_VALUES, _workload='/h/workloads/q18a-native41-progress.json')


def arm_argv(**values):
    (args,) = [a for m, a in body_commands(queue.derive()[DRAFT]) if m == 'e3_alpha_perf_box' and a.startswith('arm ')]
    values = dict(QUEUE_VALUES, **values)
    values['_label'] = '%s-%s-%s' % (values['_mode'], values['_cohort'], values['_pair'])
    return held.expand(args, values)


def scale_workload(tmp, name):
    """Test-scale manifest: the frozen file with only sizes and times reduced (leaf); the standing budget stays declared."""
    data = json.loads((HERE / 'workloads' / (name + '.json')).read_text())
    data['decode'].update(prompt_tokens=64, generation_budget=2000, open_timeout_s=10.0)
    data['prefill']['fresh']['prompt_tokens'] = 600
    data['prefill']['append']['prompt_tokens'] = 64
    data['timeouts'] = dict(controller_s=10.0, read_s=10.0, drain_s=5.0, join_s=5.0, finish_s=10.0)
    data['window_s'] = 0.5
    path = Path(tmp) / (name + '-fixture.json')
    path.write_text(json.dumps(data))
    return path


class Arm(unittest.TestCase):
    setUp = hf.Arm.setUp
    tearDown = hf.Arm.tearDown
    latch = hf.Arm.latch

    def run_arm(self, mode, cohort, index, name):
        label = progress.label(mode, cohort, index)
        out = Path(self.tmp.name) / ('arm-%s.json' % label)
        argv = arm_argv(PROBE_URL=self.server.url, EPOCH=str(self.driver.engine_epoch), _mode=mode, _cohort=cohort,
                        _pair=str(index), Q=self.tmp.name, H=str(HERE),
                        _workload=str(scale_workload(self.tmp.name, name)))
        mark, served = len(self.server.session_actions), len(self.server.completions)
        with mock.patch.object(progress.shared, '_package_gate', return_value={'ok': True}), \
                mock.patch.object(progress.base, 'SETTLE_S', 0.2), mock.patch.object(progress.base, 'CLOSE_TIMEOUT_S', 1.0), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = ap.main(argv)
        return (rc, json.loads(out.read_text()), self.server.session_actions[mark:], self.server.completions[served:])

    def test_both_bands_through_the_exact_cli_send_the_declared_budget(self):
        salts = {}
        for mode, cohort, index, name, budget, band in (
                ('burst', 'solo', 2, 'q18a-alpha-perf-band2', 32768, 2),
                ('burst', 'live', 2, 'q18a-alpha-perf-band2', 32768, 2),
                ('burst', 'live', 0, 'q18a-native41-progress', 16384, 1),
                ('current', 'solo', 1, 'q18a-native41-progress', 16384, 1)):
            with self.subTest(mode=mode, cohort=cohort, index=index):
                self.latch()
                rc, record, actions, served = self.run_arm(mode, cohort, index, name)
                self.assertEqual(rc, 2)      # No native chunk here, by construction: incomplete by its own receipts.
                self.assertNotIn('error', record, record.get('error'))
                self.assertEqual((record['label'], record['band'], record['workload']),
                                 (progress.label(mode, cohort, index), band, name))
                self.assertEqual(actions[:5], hf.ARM_ACTIONS)
                self.assertEqual(set(actions), set(hf.ARM_ACTIONS))
                self.assertEqual(record['mode_switch']['mode'], mode)
                self.assertEqual(self.progress.mode, mode)
                self.assertEqual([c['max_tokens'] for c in served], [budget] * 8 + [1, 1])
                self.assertEqual(record['standing_generation_budget'], budget)
                self.assertIn('progress-chunks-not-recorded', record['reasons'])
                self.assertEqual(record['solo_cancel']['triggered'], cohort == 'solo')
                salts[record['label']] = record['salt_sha256']
        self.assertEqual(len(set(salts.values())), 4)

    def test_the_wrong_manifest_for_the_round_is_refused_before_any_traffic(self):
        for mode, cohort, index, name in (('burst', 'live', 2, 'q18a-native41-progress'),
                                          ('burst', 'live', 0, 'q18a-alpha-perf-band2')):
            with self.subTest(index=index, manifest=name):
                rc, record, actions, served = self.run_arm(mode, cohort, index, name)
                self.assertEqual((rc, record['reasons']), (2, ['arm-refused']))
                self.assertEqual(record['error']['error_message'], 'alpha-perf-workload-declaration-invalid')
                self.assertEqual((actions, served), ([], []))
                Path(self.tmp.name, 'arm-%s.json' % record['label']).unlink()

    def test_rounds_outside_the_declared_set_do_not_parse(self):
        argv = arm_argv(_mode='burst', _cohort='live', _pair='6', Q=self.tmp.name)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            ap.main(argv)
        self.assertFalse(Path(self.tmp.name, 'arm-burst-live-6.json').exists())


# --- grade ------------------------------------------------------------------------

def memory_line(rank, txn, start, end, band):
    allocated, reserved = GPU['band%d' % band]
    free = 2 * 2 ** 20 if band == 2 else 30 * 2 ** 20
    return ('flash-s5 memory point=open rank=%d txn_seq=%d chunk=%d-%d allocated=%d reserved=%d inactive_split=%d '
            'alloc_retries=%d ooms=0 free=%d total=25296044032' % (rank, txn, start, end, allocated, reserved, 2 ** 26,
                                                                  5 + txn, free))


def synthetic_run(order=None, extra=None):
    """The declared arms in order with real emitter lines, memory receipts and, in band 2 live, recovered retries."""
    emit, arms, extra = hf.Emit(), [], extra or {}
    for mode, cohort, index in (order or ap.ARM_ORDER):
        name, band = progress.label(mode, cohort, index), ap.band_of_index(index)
        cuts = band_cuts(band)
        start = emit.tick(5_000_000)
        emit.tick(500_000)
        chunks = len(cuts) - 1
        lives = [8] * chunks if cohort == 'live' else [8] + [0] * (chunks - 1)
        for c, live in enumerate(lives):
            for rank in range(4):
                emit.log(emit.now, rank, memory_line(rank, c + 1, cuts[c], cuts[c + 1], band))
            emit.chunk(cuts[c], cuts[c + 1], mode=mode, live=live, segments=3,
                       waits=2 if (mode, cohort) == ('burst', 'live') else 0)
            if band == 2 and cohort == 'live' and c == 1:
                emit.log(emit.now, 0, 'WARNING memory allocation failed with OOM on device 0 while trying to allocate '
                                      '%d bytes' % (155 * 2 ** 20))
        if name in extra:
            emit.log(emit.now, 1, extra[name])
        end = emit.tick(1_000_000)
        window = {'aggregate_tokens_per_s': 340.0, 'stream_rates_tokens_per_s': [42.5] * 8, 'gap_ms_p99': 60.0,
                  'gap_ms_max': 100.0}
        client = {'prefill_tokens_per_s': 900.0, 'all_open_share_of_span': 1.0 if cohort == 'live' else 0.1}
        if cohort == 'live':
            client['decode_window'] = window
        arms.append(dict(label=name, mode=mode, cohort=cohort, index=index, band=band, complete=True, reasons=[],
                         wall_start_s=emit.wall(start), wall_end_s=emit.wall(end),
                         progress_delta={'chunks': chunks, 'commands': chunks * 5},
                         fresh={'prompt_tokens_observed': cuts[-1] + 1, 'ttft_s': float(chunks)},
                         decode_only={'aggregate_tokens_per_s': 400.0, 'stream_rates_tokens_per_s': [50.0] * 8},
                         client_basis=client, append={'ttft_s': 2.0}, metrics={}))
    return emit, arms


PASS = {'verdict': 'PASS'}
ALIVE = {'verdict': 'CLIENT_EXIT'}
CHECKS = hf.CHECKS


def write_streams(q, lines):
    for stream in ('log', 'err'):
        (Path(q) / ('engine-stream.' + stream)).write_text(''.join(t for s, t in lines if s == stream))


class Grade(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def grade(self, emit, arms, ingress=None, watches=None, not_run=(), lines=None):
        q = Path(tempfile.mkdtemp(dir=self.tmp.name))
        write_streams(q, emit.lines if lines is None else lines)
        ingress = {a['label']: PASS for a in arms} if ingress is None else ingress
        watches = {a['label']: ALIVE for a in arms} if watches is None else watches
        return ap.grade(arms, ingress, CHECKS, [str(q / 'engine-stream.log'), str(q / 'engine-stream.err')], not_run,
                        watches)

    def assertInconclusive(self, result, *items):
        self.assertEqual(result['verdict'], 'INCONCLUSIVE', (result['missing'], result['failures']))
        for item in items:
            self.assertIn(item, result['missing'])

    def test_memory_receipt_regex_reads_the_recorded_line(self):
        # Verbatim from held-f run 2's engine stream.
        line = ('flash-s5 memory point=open rank=0 txn_seq=1 chunk=0-1024 allocated=22696461824 reserved=23337107456 '
                'inactive_split=124746240 alloc_retries=5 ooms=0 free=693829632 total=25296044032')
        row = ap.device_memory([(None, line)])['0']
        self.assertEqual((row['open_receipts'], row['max_alloc_retries'], row['max_ooms']), (1, 5, 0))
        self.assertEqual((row['max_reserved_mib'], row['min_free_mib']), (22256.0, 661.7))

    def test_sixteen_arms_pass_with_the_table_per_arm_and_band(self):
        emit, arms = synthetic_run()
        result = self.grade(emit, arms)
        self.assertEqual((result['verdict'], result['missing'], result['failures']), ('PASS', [], []), result['missing'])
        self.assertEqual(sorted(result['table']), sorted(LABELS))
        self.assertEqual((result['required_arms'], result['optional_arms']), (LABELS[:12], LABELS[12:]))
        self.assertEqual({v['verdict'] for v in result['validity'].values()}, {'PASS'})
        self.assertEqual({(v['crash_free_completion'], v['tp_agreement'], v['disagree_lines'])
                          for v in result['validity'].values()}, {('PASS', 'PASS', 0)})
        row = result['table']['burst-live-2']
        self.assertEqual((row['band'], row['mode'], row['cohort'], row['prompt_tokens'], row['chunks']),
                         (2, 'burst', 'live', 131094, 5))
        self.assertEqual((row['ttft_s'], row['s_per_chunk'], row['prefill_tok_s']), (5.0, 1.0, 900.0))
        self.assertEqual((row['decode_during_tok_s'], row['decode_only_tok_s'], row['gap_p99_ms'], row['gap_max_ms']),
                         (340.0, 400.0, 60.0, 100.0))
        self.assertAlmostEqual(row['decode_ratio'], 0.85)
        self.assertAlmostEqual(row['min_stream_ratio'], 0.85)
        self.assertEqual(sorted(row['chunk_breakdown'], key=lambda b: int(b.split('-')[0])),
                         ['0-16384', '97152-114688', '114688-131072', '131072-197632'])
        self.assertEqual(row['commands'], 25)
        self.assertEqual({k: row['device_memory']['0'][k] for k in ('open_receipts', 'max_ooms', 'max_reserved_mib',
                                                                    'min_free_mib')},
                         dict(open_receipts=5, max_ooms=0, max_reserved_mib=23300.0, min_free_mib=2.0))
        self.assertEqual((row['allocator_retry_warnings'], row['max_retry_request_mib']), (1, 155.0))
        band2 = result['summary']['band2']
        self.assertEqual(sorted(band2['groups']), ['burst-live', 'burst-solo'])
        self.assertEqual(band2['groups']['burst-live']['arms'], ['burst-live-2', 'burst-live-3', 'burst-live-4',
                                                                 'burst-live-5'])
        self.assertIn('97152-114688', band2['groups']['burst-solo']['history_bands'])
        self.assertEqual(band2['allocator_retry_warnings'], 4)
        self.assertEqual(sorted(result['summary']['band1']['groups']), ['burst-live', 'burst-solo', 'current-live',
                                                                        'current-solo'])
        self.assertEqual(result['summary']['band1']['device_memory']['2']['min_free_mib'], 30.0)
        self.assertEqual(result['summary']['band1']['allocator_retry_warnings'], 0)
        self.assertIs(result['speed_is_gate'], False)
        self.assertEqual(result['not_run_by_budget'], [])

    def test_speed_never_changes_the_verdict(self):
        emit, arms = synthetic_run()
        slow = [dict(a, client_basis=dict(a['client_basis'], prefill_tokens_per_s=1.0)) for a in arms]
        self.assertEqual(self.grade(emit, slow)['verdict'], 'PASS')

    def test_the_budget_skip_changes_no_verdict_but_a_required_arm_does(self):
        order = ap.ARM_ORDER[:12]
        emit, arms = synthetic_run(order)
        optional = LABELS[12:]
        result = self.grade(emit, arms, not_run=optional)
        self.assertEqual((result['verdict'], result['not_run_by_budget']), ('PASS', sorted(optional)), result['missing'])
        self.assertEqual({result['validity'][x]['verdict'] for x in optional}, {'NOT_RUN'})
        self.assertInconclusive(self.grade(emit, arms), 'optional-arm-neither-run-nor-skipped-by-budget:burst-solo-5')
        # A required arm is never skipped by the budget.
        emit, arms = synthetic_run(ap.ARM_ORDER[:11])
        result = self.grade(emit, arms, not_run=optional + ['burst-solo-3'])
        self.assertInconclusive(result, 'declared-arm-missing:burst-solo-3', 'undeclared-budget-skip:burst-solo-3')

    def test_a_disagree_line_fails_that_arm_and_only_that_arm(self):
        emit, arms = synthetic_run(extra={'burst-live-2': '(Worker_TP1 pid=202) ERROR local QSA layers disagree on the '
                                                          'packed selection width'})
        result = self.grade(emit, arms)
        self.assertEqual((result['verdict'], result['failures']), ('FAIL', ['burst-live-2:tp-disagree']))
        row = result['validity']['burst-live-2']
        self.assertEqual((row['verdict'], row['tp_agreement'], row['disagree_lines'], row['crash_free_completion']),
                         ('FAIL', 'FAIL', 1, 'PASS'))
        self.assertEqual({v['verdict'] for k, v in result['validity'].items() if k != 'burst-live-2'}, {'PASS'})
        self.assertEqual(result['summary']['band2']['arm_verdicts']['burst-live-2'], 'FAIL')
        # Between the arm spans it fails the run, and the arm rows stay PASS.
        emit, arms = synthetic_run()
        gap = arms[3]['wall_end_s'] + progress.LOG_LAG_S + 0.5
        mono = emit.mono0 + int(round((gap - hf.WALL0) * 1e9))
        emit.log(mono, 1, 'ERROR cached request arrays disagree')
        result = self.grade(emit, arms)
        self.assertEqual((result['verdict'], result['failures'], result['disagree_lines_outside_every_arm']),
                         ('FAIL', ['tp-disagree-outside-every-arm-span'], 1))
        self.assertEqual({v['verdict'] for v in result['validity'].values()}, {'PASS'})

    def test_engine_watch_and_incomplete_arm_rows(self):
        emit, arms = synthetic_run()
        name = 'burst-live-3'
        watches = {a['label']: ALIVE for a in arms}
        result = self.grade(emit, arms, watches=dict(watches, **{name: {'verdict': 'DEAD'}}))
        self.assertEqual((result['verdict'], result['failures']), ('FAIL', [name + ':engine-dead']))
        self.assertEqual((result['validity'][name]['verdict'], result['validity'][name]['crash_free_completion']),
                         ('FAIL', 'FAIL'))
        # A watch stopped the client: no arm record, the row is still explicit.
        kept = [a for a in arms if a['label'] != name]
        result = self.grade(emit, kept, watches=dict(watches, **{name: {'verdict': 'DEAD'}}))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn(name + ':engine-dead', result['failures'])
        self.assertEqual((result['validity'][name]['verdict'], result['validity'][name]['arm_reasons']),
                         ('FAIL', ['no-arm-record']))
        # A missing watch record is not a pass; nor is an incomplete arm.
        watches.pop('burst-solo-3')
        self.assertInconclusive(self.grade(emit, arms, watches=watches), 'burst-solo-3:engine-watch-record-missing')
        changed = [dict(a, complete=False, reasons=['fresh-not-complete']) if a['label'] == 'burst-solo-2' else a
                   for a in arms]
        result = self.grade(emit, changed)
        self.assertInconclusive(result, 'burst-solo-2:fresh-not-complete')
        self.assertEqual((result['validity']['burst-solo-2']['verdict'],
                          result['validity']['burst-solo-2']['crash_free_completion']), ('FAIL', 'FAIL'))

    def test_held_f_completeness_defects_hold_in_band_2(self):
        emit, arms = synthetic_run()
        span = next(a for a in arms if a['label'] == 'burst-solo-2')
        inside = [i for i, (s, t) in enumerate(emit.lines) if s == 'err' and progress.PROGRESS_LINE in t
                  and span['wall_start_s'] <= progress.docker_stamp(t) <= span['wall_end_s']]
        lines = [x for i, x in enumerate(emit.lines) if i != inside[1]]
        result = self.grade(emit, arms, lines=lines)
        self.assertInconclusive(result, 'progress-burst-solo-2:worker-rows-missing')
        self.assertEqual(result['validity']['burst-solo-2']['verdict'], 'FAIL')
        # The chunk counter says more than the log holds.
        changed = [dict(a, progress_delta={'chunks': 6}) if a['label'] == 'burst-live-2' else a for a in arms]
        self.assertInconclusive(self.grade(emit, changed),
                                'progress-burst-live-2:progress-lines-differ-from-the-chunk-counter')
        # A band field that is not the round's, an arm span into the next arm, a failed ingress.
        changed = [dict(a, band=1) if a['label'] == 'burst-live-2' else a for a in arms]
        self.assertInconclusive(self.grade(emit, changed), 'burst-live-2:band-not-the-round-index-band')
        changed = [dict(a, wall_end_s=arms[i + 1]['wall_end_s']) if i == 9 else a for i, a in enumerate(arms)]
        self.assertInconclusive(self.grade(emit, changed), 'progress:engine-line-in-two-arm-spans')
        result = self.grade(emit, arms, ingress=dict({a['label']: PASS for a in arms}, **{'burst-live-3': {'verdict': 'FAIL'}}))
        self.assertEqual((result['verdict'], result['failures']), ('FAIL', ['ingress-burst-live-3:INVALID']))

    def test_grade_cli_files_rcs_and_no_overwrite(self):
        emit, arms = synthetic_run()
        q = Path(self.tmp.name)
        write_streams(q, emit.lines)
        args = []
        for a in arms:
            for prefix, data, flag in (('arm', a, None), ('ingress', PASS, '--ingress'), ('watch', ALIVE, '--watch')):
                path = q / ('%s-%s.json' % (prefix, a['label']))
                path.write_text(json.dumps(data))
                args += ['--arm', str(path)] if flag is None else [flag, '%s=%s' % (a['label'], path)]
        for name in CHECKS:
            (q / (name + '.json')).write_text(json.dumps(PASS))
            args += ['--check', '%s=%s' % (name, q / (name + '.json'))]
        args += ['--log', str(q / 'engine-stream.log'), '--log', str(q / 'engine-stream.err')]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(ap.main(['grade'] + args + ['--out', str(q / 'grade.json')]), 0)
        self.assertEqual(out.getvalue().strip(), 'ALPHA_PERF_GRADE PASS')
        self.assertEqual(json.loads((q / 'grade.json').read_text())['schema'], ap.GRADE_SCHEMA)
        self.assertEqual(ap.main(['grade'] + args + ['--out', str(q / 'grade.json')]), 2)
        # A retained xz stream reads the same: progress lines and the memory receipts both.
        import lzma
        with lzma.open(q / 'engine-stream.log.xz', 'wt') as handle:
            handle.write((q / 'engine-stream.log').read_text())
        view = ap.read_stamped([q / 'engine-stream.log.xz'], ap.is_tracked)
        self.assertEqual(view, ap.read_stamped([q / 'engine-stream.log'], ap.is_tracked))
        self.assertTrue(view)
        # An unreadable arm record stays a row.
        (q / 'arm-burst-solo-5.json').write_text('{not json')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ap.main(['grade'] + args + ['--out', str(q / 'grade-2.json')]), 2)
        self.assertIn('declared-arm-missing:burst-live-5', json.loads((q / 'grade-2.json').read_text())['missing'] +
                      ['declared-arm-missing:burst-live-5'])


# --- the derived arm loop under real bash ------------------------------------------

STUB_CHECKS = r'''#!/usr/bin/env python3
import json, os, sys
out = sys.argv[sys.argv.index('--out') + 1]
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(dict(script='checks', action=sys.argv[1], out=os.path.basename(out))) + '\n')
json.dump({'verdict': 'PASS'}, open(out, 'x'))
'''
STUB_BOX = r'''#!/usr/bin/env python3
import json, os, pathlib, shutil, sys, time
if sys.argv[1] == 'grade':
    sys.path.insert(0, os.environ['REAL_HARNESS'])
    import e3_alpha_perf_box as real
    raise SystemExit(real.main(sys.argv[1:]))
a = sys.argv
out = a[a.index('--out') + 1]
label = '%s-%s-%s' % (a[a.index('--mode') + 1], a[a.index('--cohort') + 1], a[a.index('--index') + 1])
env = {k: os.environ.get(k) for k in ('READINESS', 'GEN', 'S5_OVERLAY', 'S5_LAUNCHER_NAME', 'PACKAGE_MANIFEST', 'GENERATION')}
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(dict(script='box', action='arm', label=label, env=env, workload=os.path.basename(a[a.index('--workload') + 1]))) + '\n')
if label == os.environ.get('DIE_IN'):
    with open(os.environ['Q'] + '/engine-stream.log', 'a') as f:
        f.write(os.environ['DEATH_LINE'])
    time.sleep(600)
shutil.copy(os.environ['ARMS'] + '/arm-%s.json' % label, out)
'''
LOOP_FUNCTIONS = ('step', 'record', 'host_crash_detected', 'engine_watched', 'native41_budget')


class ArmLoop(unittest.TestCase):
    """queue.BODY runs as written; arm and check clients are stubs (leaf arm-loop-clients), the watch and grade real."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='alpha-perf-loop-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q, self.h, self.bin, self.arms = (self.root / x for x in ('q', 'h', 'bin', 'arms'))
        for d in (self.q, self.h, self.bin, self.arms):
            d.mkdir()
        (self.h / 'native43_engine_death.py').symlink_to(HERE / 'native43_engine_death.py')
        (self.h / 'alpha_perf_gate.py').symlink_to(HERE / 'alpha_perf_gate.py')
        (self.h / 'native39_trunk_gate.py').symlink_to(HERE / 'native39_trunk_gate.py')
        (self.h / 'native40_box_gate.py').symlink_to(HERE / 'native40_box_gate.py')
        (self.h / 'native41_box_gate.py').symlink_to(HERE / 'native41_box_gate.py')
        (self.h / 'e3_native39_trunk_checks.py').write_text(STUB_CHECKS)
        (self.h / 'e3_alpha_perf_box.py').write_text(STUB_BOX)
        self.tool('nvidia-smi', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')
        self.tool('docker', 'case "$*" in *State.Running*) echo true;; *) echo 2026-11-01T10:48:09.000000000Z;; esac')
        self.tool('systemctl', 'echo 0')
        (self.q / 'unarmed-stock-prefill.json').write_text(json.dumps(PASS))

    def tool(self, name, body):
        path = self.bin / name
        path.write_text('#!/bin/bash\n' + body + '\n')
        path.chmod(0o755)

    def run_loop(self, order, *, jump_after=None, die_in=None, timeout=240):
        emit, arms = synthetic_run(order)
        write_streams(self.q, emit.lines)
        for a in arms:
            (self.arms / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
        draft = queue.derive()[DRAFT]
        self.assertIn(queue.BODY, draft)
        script = ['set -uo pipefail', 'FAILS=0; INCS=0; declare -A PHASE_VERDICT=()',
                  "ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=''; CONTAINER=qwen-flash-prod; SERVICE=vllm-flash; N0=0",
                  'Q=%s; H=%s; PROBE_URL=http://127.0.0.1:1; MODEL=m; RUN_ID=r; EPOCH=1; _cid=c; IDLE_TIMEOUT_S=1' % (self.q, self.h),
                  'Q18A_READINESS=/r/ready; S5_OVERLAY=/o; S5_LAUNCHER_NAME=l.sh', 'health_ok() { return 0; }']
        script += [le_fix_function(draft, name) for name in LOOP_FUNCTIONS]
        if jump_after:
            # The box clock past the budget right after the named step (bash SECONDS is writable): leaf clock.
            script += ['eval "$(declare -f record | sed \'1s/record/real_record/\')"',
                       'record() { real_record "$@"; local rc=$?; [[ $1 == %s ]] && SECONDS=$((LAST_ARM_ENTRY_S + 10)); '
                       'return $rc; }' % jump_after]
        script += [queue.BODY, 'echo "END FAILS=$FAILS INCS=$INCS CRASHED=$ENGINE_CRASHED SOURCE=$ENGINE_CRASH_SOURCE"']
        path = self.root / 'loop.sh'
        path.write_text('\n'.join(script) + '\n')
        env = {k: v for k, v in os.environ.items() if k != 'HARNESS_DIR'}
        env.update(PATH='%s:%s' % (self.bin, os.environ['PATH']), Q=str(self.q), CALLS=str(self.root / 'calls.jsonl'),
                   ARMS=str(self.arms), REAL_HARNESS=str(HERE), DEATH_LINE=DEATH_LINE, PYTHONDONTWRITEBYTECODE='1')
        if die_in:
            env['DIE_IN'] = die_in
        proc = subprocess.Popen(['bash', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                                start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
        calls = [json.loads(x) for x in (self.root / 'calls.jsonl').read_text().splitlines()]
        steps = {m.group(1): m.group(2) for m in re.finditer(r'^STEP (\S+) (\S+)', out, flags=re.M)}
        return out, err, calls, steps

    def grade_file(self):
        return json.loads((self.q / 'alpha-perf-grade.json').read_text())

    def test_sixteen_arms_in_order_with_the_band_manifest_and_env_then_the_real_grade(self):
        out, err, calls, steps = self.run_loop(None)
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, (out[-1500:], err[-800:]))
        arm_calls = [c for c in calls if c['script'] == 'box']
        self.assertEqual([c['label'] for c in arm_calls], LABELS)
        for c in arm_calls:
            expected = 'q18a-native41-progress.json' if c['label'][-1] in '01' else 'q18a-alpha-perf-band2.json'
            self.assertEqual(c['workload'], expected, c['label'])
            # The env prefix on the shell function reaches the client through watch and timeout.
            self.assertEqual(c['env'], dict(READINESS='/r/ready', GEN='/o/S5-OVERLAP-GENERATION.json', S5_OVERLAY='/o',
                                            S5_LAUNCHER_NAME='l.sh', PACKAGE_MANIFEST='/o/SHA256SUMS',
                                            GENERATION='/o/S5-OVERLAP-GENERATION.json'))
        self.assertEqual([c['out'] for c in calls if c['script'] == 'checks' and c['action'] == 'full-idle'],
                         ['entry-%s.json' % x for x in LABELS] + ['end-full-idle.json'])
        for name in LABELS:
            self.assertEqual(steps['arm-' + name], 'PASS')
            watch = json.loads((self.q / ('engine-watch-arm-%s.json' % name)).read_text())
            self.assertEqual((watch['verdict'], watch['client_rc'], watch['container']), ('CLIENT_EXIT', 0, 'qwen-flash-prod'))
            self.assertTrue((self.q / ('engine-watch-entry-%s.json' % name)).is_file())
        self.assertEqual(steps['alpha-perf'], 'PASS')
        result = self.grade_file()
        self.assertEqual((result['verdict'], result['missing']), ('PASS', []))
        self.assertEqual({v['engine_watch'] for v in result['validity'].values()}, {'CLIENT_EXIT'})

    def test_budget_past_the_required_arms_skips_only_optional_and_changes_no_verdict(self):
        out, err, calls, steps = self.run_loop(ap.ARM_ORDER[:12], jump_after='ingress-burst-solo-3')
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, (out[-1500:], err[-800:]))
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box'], LABELS[:12])
        for name in LABELS[12:]:
            self.assertEqual(steps['arm-' + name], 'NOT_RUN')
            self.assertFalse((self.q / ('entry-%s.json' % name)).exists())
        result = self.grade_file()
        self.assertEqual((result['verdict'], result['not_run_by_budget']), ('PASS', sorted(LABELS[12:])), result['missing'])
        self.assertEqual(steps['alpha-perf'], 'PASS')

    def test_budget_before_a_required_arm_is_inconclusive(self):
        out, err, calls, steps = self.run_loop(ap.ARM_ORDER[:10], jump_after='ingress-burst-live-2')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box'], LABELS[:10])
        self.assertEqual((steps['arm-burst-live-3'], steps['arm-burst-solo-3'], steps['arm-burst-solo-4']),
                         ('INCONCLUSIVE', 'INCONCLUSIVE', 'NOT_RUN'))
        self.assertIn('END FAILS=0 INCS=3 CRASHED=0', out, out[-800:])       # the two arms and the grade
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('declared-arm-missing:burst-live-3', result['missing'])

    def test_a_dead_engine_stops_the_band_2_arm_and_keeps_band_1(self):
        started = time.monotonic()
        out, err, calls, steps = self.run_loop(ap.ARM_ORDER[:9], die_in='burst-live-2')
        self.assertLess(time.monotonic() - started, 120)
        self.assertIn('CRASHED=1 SOURCE=engine-death', out, out[-800:])
        self.assertEqual(steps['arm-burst-live-2'], 'FAIL')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box'], LABELS[:10])
        for name in LABELS[10:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
        watch = json.loads((self.q / 'engine-watch-arm-burst-live-2.json').read_text())
        self.assertEqual(watch['verdict'], 'DEAD')
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('burst-live-2:engine-dead', result['failures'])
        self.assertEqual(result['validity']['burst-live-2']['verdict'], 'FAIL')
        self.assertEqual({result['validity'][x]['verdict'] for x in LABELS[:9]}, {'PASS'})
        self.assertEqual(sorted(result['summary']['band1']['arm_verdicts']), sorted(LABELS[:8]))


def le_fix_function(text, name):
    lines = text.splitlines(keepends=True)
    start = [i for i, line in enumerate(lines) if line == name + '() {\n']
    assert len(start) == 1, (name, start)
    end = next(i for i in range(start[0] + 1, len(lines)) if lines[i] == '}\n')
    return ''.join(lines[start[0]:end + 1])


# --- commands, derivation, window -----------------------------------------------------

ARGPARSE_VALUES = dict(held.ARGPARSE_VALUES, QNAME='q18a-alpha-perf', RUN_ID='q18a-alpha-perf-fixture', _mode='burst',
                       _cohort='live', _pair='2', _label='burst-live-2', _workload='/h/workloads/q18a-alpha-perf-band2.json', name='x')


def bash_words(args, draft, values=None):
    """Expand one call's argument text with the real bash (set -u), the derived settings sourced."""
    values = dict(ARGPARSE_VALUES, **(values or {}))
    budget = le_fix_function(draft, 'native41_budget')
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(HERE))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    lines += ['GRADE_ARGS=(--arm "$Q/arm-burst-live-2.json" --ingress "burst-live-2=$Q/ingress-burst-live-2.json" '
              '--watch "burst-live-2=$Q/engine-watch-arm-burst-live-2.json")',
              'source "$H/runs/q18a_alpha_perf_settings.sh" || exit 97', budget, "printf '%s\\0' " + args]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode().split('\0')[:-1], done.stderr.decode()[-400:]


class Commands(unittest.TestCase):
    def test_every_body_command_parses_with_its_module(self):
        import e3_native39_trunk_checks as checks
        modules = {'e3_alpha_perf_box': ap, 'e3_native39_trunk_checks': checks}
        rows = body_commands(queue.derive()[DRAFT])
        self.assertEqual(sorted({m for m, _ in rows}), sorted(modules))
        actions = []
        original = argparse.ArgumentParser.parse_args

        def capture(parser, args=None, namespace=None):
            raise held.Parsed(original(parser, args, namespace))
        for module, args in rows:
            argv = held.expand(args, dict(QUEUE_VALUES, _mode='burst', _cohort='live', _pair='2', _label='burst-live-2',
                                          _workload='/h/workloads/q18a-alpha-perf-band2.json'))
            with mock.patch.object(argparse.ArgumentParser, 'parse_args', capture):
                with self.assertRaises(held.Parsed) as parsed, contextlib.redirect_stderr(io.StringIO()):
                    modules[module].main(argv)
            actions.append((module, argv[0], vars(parsed.exception.args[0])))
        self.assertEqual([a[1] for a in actions], ['instruments', 'full-idle', 'arm', 'ingress', 'instruments',
                                                  'full-idle', 'grade'])
        arm = next(ns for m, a, ns in actions if a == 'arm')
        self.assertEqual((arm['mode'], arm['cohort'], arm['index'], arm['control_timeout_s']), ('burst', 'live', 2, 60.0))
        self.assertTrue(arm['workload'].endswith('workloads/q18a-alpha-perf-band2.json'))
        grade = actions[-1][2]
        self.assertEqual(grade['out'], '/q/alpha-perf-grade.json')
        self.assertEqual(grade['log'], ['/q/engine-stream.log', '/q/engine-stream.err'])
        ingress = next(ns for m, a, ns in actions if a == 'ingress')
        self.assertEqual(ingress['generation'], progress.base.ARM_GENERATIONS)

    def test_the_loop_list_is_the_declared_order_and_the_split_runs_in_bash(self):
        draft = queue.derive()[DRAFT]
        (loop,) = re.findall(r'^\s*for _label in ([a-z0-9 -]+); do$', draft, flags=re.M)
        self.assertEqual(loop.split(), LABELS)
        self.assertEqual(list(queue.ARMS), LABELS)
        self.assertEqual(LABELS[:8], [progress.label(*x) for x in progress.ARM_ORDER])
        self.assertEqual(queue.ARMS_BAND2_REQUIRED + queue.ARMS_BAND2_OPTIONAL, tuple(BAND2_LABELS))
        split = '    IFS=- read -r _mode _cohort _pair <<< "$_label"'
        self.assertIn(split, draft)
        script = 'for _label in %s; do\n%s\nprintf "%%s %%s %%s\\n" "$_mode" "$_cohort" "$_pair"\ndone' % (loop, split)
        done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
        self.assertEqual([tuple(x.split()) for x in done.stdout.splitlines()], [(m, c, str(i)) for m, c, i in ap.ARM_ORDER])
        # Band 2 keeps solo first in each round of the repeat, so a live failure keeps its solo number.
        self.assertEqual(BAND2_LABELS[:2], ['burst-solo-2', 'burst-live-2'])
        self.assertEqual({x.split('-')[0] for x in BAND2_LABELS}, {'burst'})


class QueueArgparse(unittest.TestCase):
    def test_every_queue_python_call_parses_with_its_real_parser(self):
        texts = queue.derive()
        draft = texts[DRAFT]
        calls = held.queue_python_calls(texts, (HERE / 'qlib.sh').read_text())
        scripts = sorted({c[1] for c in calls})
        self.assertEqual(scripts, sorted([
            'SETCFG', 'e2_memory_attribution.py', 'e3_native28_host_observer.py', 'e3_native30_e_grade.py',
            'e3_native39_trunk_checks.py', 'e3_native39_w2_length.py', 'e3_alpha_perf_box.py',
            'e3_native41_unarmed_box.py', 'e3_native_speculation_gate.py', 'e3_native_startup_grade.py',
            'e3_private_auth.py', 'engine_identity.py', 'native35_ledger_grade.py', 'native43_engine_death.py',
            'native43_allocator_phase_deltas.py']))
        setcfg = next(p for p in (ROOT / 'deploy/setcfg.py', HERE / 'setcfg.reference.py') if p.is_file())
        rows = []
        for name, script, args, form in calls:
            with self.subTest(file=name.split('/')[-1], script=script, args=args[:60]):
                rc, words, err = bash_words(args, draft)
                self.assertEqual(rc, 0, err)
                if form != 'python3':
                    words = held.QLIB_CALLS[form][1](words)
                if script == 'SETCFG':
                    verdict, detail = held.setcfg_parse(setcfg, words)
                else:
                    verdict, detail = held.real_parse(HERE / script, words)
                self.assertEqual(verdict, 'parsed', (script, words[:4], detail))
                rows.append(dict(file=name.split('/')[-1], script=script, form=form,
                                 action=words[0] if words and not words[0].startswith('-') else None,
                                 words=len(words), verdict=verdict))
        self.assertEqual(len(rows), len(calls))
        # Held-f's calls: the same body size, one more length guard, the watch and the M3 receipt (the death scan runs
        # as "${H:-}/...", exercised for real by ArmLoop).
        parent = hf.queue.derive()
        parent_calls = held.queue_python_calls(parent, (HERE / 'qlib.sh').read_text())
        self.assertEqual(len(rows), len(parent_calls) + 3)
        ARGPARSE_RECEIPTS[:] = rows
        guard = [a for n, s, a, f in calls if s == 'e3_native39_w2_length.py']
        self.assertEqual(len(guard), 2)
        self.assertIn('--workload "$H/workloads/q18a-native41-progress.json"', guard[0])
        self.assertIn('--workload "$H/workloads/q18a-alpha-perf-band2.json"', guard[1])

        def first(script, action):
            return next(a for n, s, a, f in calls if s == script and a.split()[0] == action)
        arm, grade = first('e3_alpha_perf_box.py', 'arm'), first('e3_alpha_perf_box.py', 'grade')
        for label, original, (old, new) in (('arm-unknown-option', arm, ('--control-timeout-s', '--control-timeout-sec')),
                                            ('arm-undeclared-mode', arm, ('--mode "$_mode"', '--mode fast')),
                                            ('arm-undeclared-round', arm, ('--index "$_pair"', '--index 6')),
                                            ('grade-without-out', grade, ('--out "$Q/alpha-perf-grade.json"', ''))):
            with self.subTest(control=label):
                self.assertEqual(original.count(old), 1)
                rc, words, err = bash_words(original.replace(old, new), draft)
                self.assertEqual(rc, 0, err)
                self.assertEqual(held.real_parse(HERE / 'e3_alpha_perf_box.py', words)[0], 'refused', label)


class Derivation(unittest.TestCase):
    def test_derived_texts_are_on_disk_parse_and_carry_the_three_flags(self):
        derived = queue.derive_all()
        for name, text in derived.items():
            self.assertEqual((ROOT / name).read_text(), text, name)
        for name in queue.derive():
            self.assertEqual(subprocess.run(['bash', '-n', str(ROOT / name)], capture_output=True).returncode, 0, name)
        settings = derived[SETTINGS]
        extra = [line for line in settings.splitlines() if line.strip().startswith('"EXTRA_ENV=')]
        self.assertEqual(len(extra), 1)
        words = extra[0].split()
        for flag in ('FLASH_S5_AUTO_ARM=1', 'FLASH_S5_NATIVE_PROGRESS=burst', 'FLASH_S5_NATIVE_QSA_POLICY=dense',
                     'FLASH_S5_REPLAY_OBSERVER=0', 'FLASH_S5_ENGINE_OFFERS=0'):
            self.assertEqual(words.count(flag), 1, flag)
        self.assertNotIn('FLASH_S5_NATIVE_QSA_TRACE', extra[0])
        self.assertNotIn('FLASH_S5_NATIVE_QSA_STOP_AFTER', extra[0])
        self.assertNotIn('FLASH_S5_UNION_RETIRE', extra[0])      # S1 retirement stays off in every arm
        # The dense spelling is the engine's own name and value.
        import qsa_operation as qo
        self.assertEqual(qo.native_qsa_settings({queue.DENSE.split('=')[0]: queue.DENSE.split('=')[1]}), ('dense', False))
        # Against held-f's settings: that one token and its comment, nothing else.
        old = (ROOT / 'harness/runs/q18a_native41_progress_settings.sh').read_text()
        old = old.replace(queue.OLD, queue.NEW).replace(queue.OLD_Q, queue.NEW_Q)
        diff = [line for line in difflib.ndiff(old.splitlines(), settings.splitlines()) if line[:1] in '+-']
        self.assertEqual([line[2:].replace(' ' + queue.DENSE, '') for line in diff if line[0] == '+' and 'EXTRA_ENV' in line],
                         [line[2:] for line in diff if line[0] == '-'])
        self.assertTrue(all(line.startswith('+ #') for line in diff if line[0] == '+' and 'EXTRA_ENV' not in line))
        draft, declare = derived[DRAFT], derived[DECLARE]
        for text, spacing in ((draft, " == "), (declare, "==")):
            self.assertIn("assert env.get('FLASH_S5_NATIVE_QSA_POLICY')%s'dense'" % spacing, text)
            self.assertIn("'FLASH_S5_NATIVE_QSA_TRACE' not in env", text)
            self.assertIn("'FLASH_S5_UNION_RETIRE' not in env", text)
            self.assertIn("assert env.get('FLASH_S5_NATIVE_PROGRESS')%s'burst'" % spacing, text)
        self.assertIn('from alpha_perf_gate import window_ok', draft)
        self.assertIn('from alpha_perf_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,PHASES', draft)
        self.assertNotIn('from native41_progress_gate import', draft)
        self.assertNotIn('python3 "$H/e3_native41_progress_box.py"', draft)
        self.assertNotIn(hf.queue.BODY, draft)
        self.assertIn(queue.GUARD_NEW, draft)
        self.assertIn(le_fix.D3_NEW, draft)
        self.assertIn(queue.m3.M3_NEW, draft)
        self.assertIn('engine_watched() {', draft)
        self.assertIn("perf=p.parent/'alpha-perf-grade.json'", draft)
        self.assertNotIn('native41-progress-grade', draft)
        self.assertLess(draft.index('--out "$Q/unarmed-stock-prefill.json"'), draft.index(queue.BODY.splitlines()[0]))
        # The engine facing clients of the body run under the watch; no tool call is uncapped.
        self.assertEqual(queue.BODY.count('engine_watched '), 3)
        self.assertEqual(len(re.findall(r'^\s*python3 "\$H/', queue.BODY, flags=re.M)), 0)
        for name in ('alpha_perf_gate.py', 'e3_alpha_perf_box.py', 'native43_engine_death.py', 'e3_native41_progress_box.py'):
            self.assertIn(name, draft)
            self.assertIn(name, declare)
            self.assertTrue((HERE / name).is_file(), name)

    def test_everything_kept_is_held_f_line_for_line(self):
        old = hf.queue.derive()[hf.DRAFT].replace(queue.OLD, queue.NEW).replace(queue.OLD_Q, queue.NEW_Q)
        old = old.replace('workloads/q18a-alpha-perf.json', 'workloads/q18a-native41-progress.json')
        new = queue.derive()[DRAFT]
        body_old = hf.queue.BODY.replace(queue.OLD, queue.NEW).replace(queue.OLD_Q, queue.NEW_Q)
        body_old = body_old.replace('workloads/q18a-alpha-perf.json', 'workloads/q18a-native41-progress.json')
        self.assertEqual(old.count(body_old), 1)
        old_rest = old.replace(body_old, '').splitlines()
        it = iter(new.replace(queue.BODY, '').splitlines())
        # Every held-f line outside the body survives in order, unless it is one of the declared swaps.
        changed = {'from native41_progress_gate import ACTIVE_S,CLEANUP_S,PHASES',
                   '# arms are replaced by the held-f arms. Every cap is native41_progress_gate.PHASES.',
                   "print(ACTIVE_S if sys.argv[1]=='active' else CLEANUP_S if sys.argv[1]=='cleanup' else PHASES[sys.argv[1]])",
                   'from native41_progress_gate import window_ok', '  e3_native41_progress_box.py native41_progress_gate.py; do',
                   '   -r $H/workloads/q18a-native41-progress.json &&'}
        old_guard = [x for x in hf.queue.GUARD_NEW.splitlines()]
        for line in old_rest:
            if line in changed or line in old_guard or 'native41-progress-grade' in line:
                continue
            self.assertIn(line, it, line[:100])


class Workload(unittest.TestCase):
    def test_band_2_is_the_held_f_manifest_with_the_declared_prompt(self):
        band1 = json.loads((HERE / queue.BAND1_MANIFEST).read_text())
        data = json.loads((HERE / queue.BAND2_MANIFEST).read_text())
        problems, cfg = progress.shared.validate_manifest(data)
        self.assertEqual((problems, cfg['repeats']), ([], 1))
        self.assertEqual(ap.standing_budget(data, 2), gate_budget := 32768)
        self.assertEqual(ap.standing_budget(band1, 1), 16384)
        for wrong in ((band1, 2), (data, 1)):
            with self.assertRaises(ValueError):
                ap.standing_budget(*wrong)
        own = ('workload', 'notes', 'progress', 'length_declaration', 'prefill')
        self.assertEqual({k: v for k, v in data.items() if k not in own}, {k: v for k, v in band1.items() if k not in own})
        self.assertEqual(data['prefill']['append'], band1['prefill']['append'])
        self.assertEqual({k: v for k, v in data['prefill']['fresh'].items() if k != 'prompt_tokens'},
                         {k: v for k, v in band1['prefill']['fresh'].items() if k != 'prompt_tokens'})
        # The counts the queue's guard checks on /tokenize, both with a native plan.
        self.assertEqual(data['length_declaration'], length.declaration(data))
        *_, fresh, append = length.counts(data['prefill']['fresh'], data['prefill']['append'])
        self.assertEqual((fresh, append), (131094, 135085))
        self.assertTrue(data['length_declaration']['fresh']['plans'] and data['length_declaration']['append']['plans'])
        for residue in (fresh % 176, append % 176):
            self.assertFalse(2 <= residue <= 64)
        # Past the arena by about 33 chunks of 1024, inside the driver's prompt limit and the history pages.
        self.assertEqual((fresh - gate_rows()) // 1024, 33)
        self.assertLess(fresh, 200000)
        self.assertLessEqual(data['decode']['prompt_tokens'] + gate_budget, progress.MODEL_LEN)

    def test_history_bands_extend_held_f_and_report_the_boundary(self):
        self.assertEqual(ap.ARENA_ROWS, 24 * 4048)
        self.assertEqual(progress.band_of(97151), '65536-97152')
        self.assertEqual(progress.band_of(97152), '97152-114688')
        self.assertEqual(progress.band_of(131072), '131072-197632')


def gate_rows():
    return ap.ARENA_ROWS


class Window(unittest.TestCase):
    def test_the_box_budget_and_the_arm_entry_rule(self):
        import native40_box_gate as n40
        import native41_box_gate as n41
        self.assertEqual((gate.CLEANUP_S, gate.NOT_BEFORE, gate.HARD_END), (n40.CLEANUP_S, n40.NOT_BEFORE, n40.HARD_END))
        for name in n41.KEPT:
            self.assertEqual(gate.PHASES[name], n40.PHASES[name], name)
        self.assertEqual(gate.ACTIVE_S + gate.CLEANUP_S, gate.BOX_BUDGET_S)
        self.assertEqual(gate.BOX_BUDGET_S, 9000)
        self.assertEqual((gate.ARMS, len(queue.ARMS), gate.BAND1_ARMS, gate.BAND2_REQUIRED_ARMS), (16, 16, 8, 4))
        self.assertEqual(gate.ARM_RESERVE_S, 450 + 1500 + 30 + 60)
        # An arm entering at the last allowed second still fits its own caps, then the whole tail, in ACTIVE_S.
        self.assertEqual(gate.LAST_ARM_ENTRY_S + gate.ARM_RESERVE_S + gate.TAIL_RESERVE_S, gate.ACTIVE_S)
        self.assertTrue(gate.arm_fits(gate.LAST_ARM_ENTRY_S))
        self.assertFalse(gate.arm_fits(gate.LAST_ARM_ENTRY_S + 1))
        # All nominal arms fit well inside the rule (the declared band 2 estimate, at twice its size for the live arms).
        per = gate.EXPECTED_RUNTIME_S['per_arm_s']
        self.assertLess(gate.EXPECTED_RUNTIME_S['nominal'], gate.BOX_BUDGET_S)
        self.assertLess(sum(per.values()) + 4 * per['band 2 burst-live'], gate.LAST_ARM_ENTRY_S)
        body = queue.BODY
        for text, count in (('timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments', 2),
                            ('timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle', 2),
                            ('timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_alpha_perf_box.py" arm', 1),
                            ('timeout 60s python3 "$H/e3_native39_trunk_checks.py" ingress', 1),
                            ('timeout 600s python3 "$H/e3_alpha_perf_box.py" grade', 1)):
            self.assertEqual(body.count(text), count, text)
        self.assertEqual((n41.INSTRUMENTS_CAP_S, n41.FULL_IDLE_CAP_S, n41.INGRESS_CAP_S, gate.GRADE_CAP_S), (120, 450, 60, 600))
        self.assertEqual(gate.latest_arm(), gate.HARD_END - gate.ACTIVE_S - gate.CLEANUP_S)
        self.assertTrue(gate.window_ok(gate.latest_arm()))
        self.assertFalse(gate.window_ok(gate.latest_arm() + 1))
        self.assertFalse(gate.window_ok(gate.NOT_BEFORE - 1))
        func = le_fix_function(queue.derive()[DRAFT], 'native41_budget')
        for key, value in (('active', gate.ACTIVE_S), ('cleanup', gate.CLEANUP_S), ('health', gate.PHASES['health']),
                           ('end_identity', gate.PHASES['end_identity']), ('last_arm_entry', gate.LAST_ARM_ENTRY_S)):
            done = subprocess.run(['bash', '-c', func + 'native41_budget ' + key],
                                  env=dict(os.environ, HARNESS_DIR=str(HERE)), capture_output=True, text=True)
            self.assertEqual((done.returncode, done.stdout.strip()), (0, str(value)), done.stderr)

    def test_draft_window_slot_executes_at_real_wall_clock(self):
        draft = queue.derive()[DRAFT]
        start = draft.index("python3 - <<'PY_SLOT'")
        slot = draft[start:draft.index('\nPY_SLOT\n', start) + len('\nPY_SLOT\n')]
        self.assertIn('alpha_perf_gate', slot)
        done = subprocess.run(['bash', '-c', 'fail_early() { return 2; }\n' + slot],
                              env=dict(os.environ, HARNESS_DIR=str(HERE)), capture_output=True, text=True)
        expected = 0 if gate.window_ok(time.time()) else 2
        self.assertEqual(done.returncode, expected, done.stderr)
        SLOT_RECEIPTS.append(dict(draft='q18a_alpha_perf.draft.sh', real_wall_clock=True,
                                  verdict='PASS' if done.returncode == 0 else 'OUTSIDE-WINDOW',
                                  active_s=gate.ACTIVE_S, cleanup_margin_s=gate.CLEANUP_S))


if __name__ == '__main__':
    unittest.main()
