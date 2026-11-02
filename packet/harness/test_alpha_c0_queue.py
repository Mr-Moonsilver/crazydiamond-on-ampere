"""C0 queue: the typed selection over the real control plane, the grade over real emitters, the arm loop in bash. CPU only.

Classes:

* Arm: the derived draft's exact arm command line through two NativeRuntimeClients over loopback HTTP into the real endpoint,
  session, control client, staged driver, AutoArmPolicy and NativeProgress of a startup-bound stack. As in held-f's Arm, nothing
  links the HTTP prompt to the scheduler, so no native chunk runs here (a real stage chunk on a connected Core is
  e3_alpha_c0_probe.py in the engine self-run). The router's refusals are test_e3_stage_local_control.py.
* Grade: eight arms of lines written by the real nm.ProgressTrace emitters (StageEmit), memory and stage receipts in the
  engine's format: verdict per arm and overall, HARNESS and ENGINE, engagement both ways, numerics, the memory comparison, the
  report's sections.
* Tears: the pre-registered rule for torn lines and lost flushes, in the shapes of the two real tears.
* ArmLoop: the derived body under real bash with the draft's own functions, the real death watch and the real grade, over stub
  arm and check clients.
* Commands / QueueArgparse: every python3 call of the derived texts parses with its own module's real parser.
* Derivation / Window: derived texts on disk and parse; the flags; everything else is alpha-perf line for line; the budget.

Fixture leaves: alpha_c0_fixture_leaves.json, beside the inherited files it names.
"""
import argparse
import contextlib
import difflib
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
import e3_alpha_c0_box as box  # noqa: E402
import alpha_c0_gate as gate  # noqa: E402
import alpha_c0_queue as queue  # noqa: E402
import engine_driver as ed  # noqa: E402
import native_auto_arm as auto  # noqa: E402
import probe_control as pc  # noqa: E402
from stage_progress import StagedSegmentDriver  # noqa: E402
import test_alpha_perf_queue as tap  # noqa: E402
import test_e3_native_overlap as native  # noqa: E402
import test_native38_plain_bind as native38  # noqa: E402
import test_native41_box_queue as held  # noqa: E402
import test_native41_progress_queue as hf  # noqa: E402

progress = box.progress
ap = box.ap
SLOT_RECEIPTS = []
ARGPARSE_RECEIPTS = []
KEY = held.KEY
DRAFT = 'harness/runs/q18a_alpha_c0.draft.sh'
SETTINGS = 'harness/runs/q18a_alpha_c0_settings.sh'
DECLARE = 'harness/runs/q18a_alpha_c0_declare.sh'
LABELS = box.required_labels()
ARM_ACTIONS = ['status', 'pause', 'native_progress', 'native_stage_local', 'release_auto_arm', 'resume']
ALL = {'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst', 'FLASH_S5_STAGE_LOCAL': '1'}
SWITCHES = tuple(ALL)
BAND1 = 'q18a-native41-progress'


def attach(engine_client, values):
    """The real attach_native_router as the API server calls it (its env is the server's)."""
    env = {'FLASH_S5_OVERLAP': '1', 'VLLM_API_KEY': KEY}
    env.update(values)
    app = held.App(engine_client)
    with held.fastapi_leaf():
        state = pc.attach_native_router(SimpleNamespace(api_key=None), app, env=env)
    (path, endpoint, methods), = app.routers[0].routes
    assert (path, methods) == (pc.NATIVE_PATH, ('POST',))
    return app, endpoint, state


def arm_argv(**values):
    """The body's exact arm command line, expanded."""
    (args,) = [a for m, a in body_commands(queue.derive()[DRAFT]) if m == 'e3_alpha_c0_box' and a.startswith('arm ')]
    values = dict(tap.QUEUE_VALUES, **values)
    values['_label'] = '%s-%s-%s' % (values['_mode'], values['_cohort'], values['_pair'])
    return held.expand(args, values)


def body_commands(text):
    body = text[text.index(queue.BODY.splitlines()[0]):]
    body = body[:body.index('record alpha-c0 ')]
    joined = re.sub(r'\\\n\s*', ' ', body)
    rows = []
    for line in joined.splitlines():
        match = re.search(r'python3 "\$H/([a-z0-9_]+)\.py"(.*)$', line)
        if match:
            rows.append((match.group(1), re.split(r'\s+>\s*"', match.group(2))[0].strip()))
    return rows


class Arm(unittest.TestCase):
    INSTALL = True

    def setUp(self):
        self._logger = sys.modules.get('vllm.logger')
        sys.modules['vllm.logger'] = SimpleNamespace(init_logger=logging.getLogger)
        driver_class = StagedSegmentDriver if self.INSTALL else ed.NativeSegmentDriver
        with mock.patch.object(ed, 'NativeSegmentDriver', driver_class):
            self.stack, self.core = native38.OwnerStartup().stack()
        held.pinned_reads(self.stack.scheduler)
        self.ready = native38.bootstrap_full_startup(self.core, self.stack.driver, timeout_s=10)
        self.driver = self.stack.driver
        # s5_startup.install_engine's order: the auto-arm policy, then progress.
        self.policy = auto.install(self.core, self.driver, {auto.ENV: '1'})
        self.progress = self.driver.install_progress('burst')
        self.env = mock.patch.dict(os.environ, dict(ALL, VLLM_API_KEY=KEY))
        self.env.start()
        self.loop = held.EngineLoop(self.core)
        app, endpoint, _ = attach(native.Frontend(self.core), ALL)
        self.server = hf.ProgressServer(self.loop, app, endpoint).__enter__()
        self.tmp = tempfile.TemporaryDirectory()

    tearDown = hf.Arm.tearDown
    latch = hf.Arm.latch

    def run_arm(self, selection, cohort, index, name=BAND1):
        label = box.label(selection, cohort, index)
        out = Path(self.tmp.name) / ('arm-%s.json' % label)
        argv = arm_argv(PROBE_URL=self.server.url, EPOCH=str(self.driver.engine_epoch), _mode=selection, _cohort=cohort,
                        _pair=str(index), Q=self.tmp.name, H=str(HERE),
                        _workload=str(tap.scale_workload(self.tmp.name, name)))
        mark, served = len(self.server.session_actions), len(self.server.completions)
        with mock.patch.object(progress.shared, '_package_gate', return_value={'ok': True}), \
                mock.patch.object(progress.base, 'SETTLE_S', 0.2), mock.patch.object(progress.base, 'CLOSE_TIMEOUT_S', 1.0), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = box.main(argv)
        return rc, json.loads(out.read_text()), self.server.session_actions[mark:], self.server.completions[served:]

    def test_both_selections_through_the_exact_cli(self):
        salts = {}
        before = progress.label, progress.release, progress.progress_reasons, progress.progress_view, progress.ACTIONS
        actions_before = progress.base.ARM_ACTIONS
        row_before, stop_before = progress.base.request_row, progress.streams._stop_streams
        for selection, cohort, index, previous in (('local', 'live', 0, True), ('serial', 'live', 0, True),
                                                   ('serial', 'solo', 1, False), ('local', 'solo', 1, False)):
            with self.subTest(selection=selection, cohort=cohort, index=index):
                self.latch()
                rc, record, actions, served = self.run_arm(selection, cohort, index)
                self.assertEqual(rc, 2)      # No native chunk here, by construction: incomplete by its own receipts.
                self.assertNotIn('error', record, record.get('error'))
                self.assertEqual((record['label'], record['selection'], record['mode'], record['band'], record['workload']),
                                 (box.label(selection, cohort, index), selection, 'burst', 1, BAND1))
                self.assertEqual(actions[:6], ARM_ACTIONS)
                self.assertEqual(set(actions), set(ARM_ACTIONS))
                self.assertEqual([actions.count(a) for a in ARM_ACTIONS[1:]], [1, 1, 1, 1, 1])
                self.assertEqual(record['selection_switch']['local'], selection == 'local')
                self.assertEqual((record['selection_switch']['previous'], record['selection_switch']['route']), (previous, 'full'))
                self.assertEqual((record['mode_switch']['mode'], self.progress.mode), ('burst', 'burst'))
                self.assertEqual((self.driver.stage_selected, record['release']['released']), (selection == 'local', True))
                self.assertEqual({r['selected'] for r in (s['native_stage_local'] for s in record['progress']
                                                          if s['label'] != 'entry')}, {selection == 'local'})
                self.assertEqual(record['stage_delta'], dict.fromkeys(box.STAGE_COUNTERS, 0))
                self.assertEqual(record['stage_delta_append'], dict.fromkeys(box.STAGE_COUNTERS, 0))
                self.assertEqual([c['max_tokens'] for c in served], [16384] * 8 + [1, 1])
                self.assertIn('progress-chunks-not-recorded', record['reasons'])
                self.assertNotIn('arm-sent-a-non-receipt-action', record['reasons'])
                self.assertNotIn('control-sequence-not-pause-progress-selection-release-resume', record['reasons'])
                self.assertFalse([x for x in record['reasons'] if x.startswith('auto-not-proven:unexpected')], record['reasons'])
                self.assertEqual(sorted(record['outputs']), ['append', 'fresh', 'standing'])
                self.assertEqual(len(record['outputs']['standing']), 8)
                self.assertTrue(all(len(r['prefix_sha256']) == 64 for r in record['outputs']['standing'].values()))
                salts[record['label']] = record['salt_sha256']
        self.assertEqual(len(set(salts.values())), 4)
        # The seams are this process's one arm; they are back after each.
        self.assertEqual((progress.label, progress.release, progress.progress_reasons, progress.progress_view, progress.ACTIONS),
                         before)
        self.assertIs(progress.base.ARM_ACTIONS, actions_before)
        self.assertEqual((progress.base.request_row, progress.streams._stop_streams), (row_before, stop_before))

    def test_rounds_and_selections_outside_the_declared_set_do_not_parse(self):
        for selection, index in (('local', '2'), ('burst', '0'), ('current', '0'), ('fused', '0')):
            argv = arm_argv(_mode=selection, _cohort='live', _pair=index, Q=self.tmp.name)
            with self.subTest(selection=selection), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                box.main(argv)
            self.assertFalse(Path(self.tmp.name, 'arm-%s-live-%s.json' % (selection, index)).exists())


class ArmNotInstalled(Arm):
    INSTALL = False

    def test_both_selections_through_the_exact_cli(self):
        self.latch()
        rc, record, actions, served = self.run_arm('local', 'solo', 0)
        self.assertEqual((rc, record['complete'], served), (2, False, []))
        self.assertIn('arm-error', record['reasons'])
        self.assertEqual(record['error']['error_message'], 'native-stage-local-not-installed')
        self.assertEqual(actions, ['status'])

    def test_rounds_and_selections_outside_the_declared_set_do_not_parse(self):
        pass


class Selection(unittest.TestCase):
    """The arm's own proofs over declared client replies (leaf arm-control-replies), each defect its row."""

    class Client:
        def __init__(self, local_reply=None, route='full', released=True, raises=None):
            self.local_reply, self.route, self.released, self.raises = local_reply, route, released, raises
            self.calls = []

        def pause(self):
            self.calls.append('pause')
            return dict(route='full', route_generation=1)

        def resume(self):
            self.calls.append('resume')
            return dict(resumed=True)

        def call(self, action, **fields):
            self.calls.append(action)
            if action == self.raises:
                raise ValueError('refused-by-the-engine')
            if action == 'native_progress':
                return dict(mode='burst', previous='current', installed='burst', route='full', route_generation=1, engine_epoch=7)
            if action == 'native_stage_local':
                local = fields['local'] if self.local_reply is None else self.local_reply
                return dict(local=local, previous=False, route=self.route, route_generation=1, engine_epoch=7)
            return dict(released=self.released, engine_epoch=7)

    def test_order_and_the_reply_is_checked(self):
        client, record = self.Client(), {}
        box.release_selected(True)(client, record, 'burst')
        self.assertEqual(client.calls, ['pause', 'native_progress', 'native_stage_local', 'release_auto_arm', 'resume'])
        for client, text in ((self.Client(local_reply=False), 'native-stage-local-selection-unproved'),
                             (self.Client(route='split'), 'native-stage-local-selection-unproved'),
                             (self.Client(released=False), 'auto-arm-release-unproved')):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, text):
                box.release_selected(True)(client, {}, 'burst')
            self.assertEqual(client.calls[-1], 'resume')

    def test_a_refused_selection_still_resumes_and_never_releases(self):
        client = self.Client(raises='native_stage_local')
        with self.assertRaisesRegex(ValueError, 'refused-by-the-engine'):
            box.release_selected(True)(client, {}, 'burst')
        self.assertEqual(client.calls, ['pause', 'native_progress', 'native_stage_local', 'resume'])

    def receipts(self, *, history=None, before=None, after=None):
        block = dict(selected=True, selections=1, chunks=0, headroom_refused=0, local_releases=0, both_stages_ns=0)
        record = dict(control_history=['status'] + ARM_ACTIONS[1:] if history is None else history, progress=[
            dict(label='before-fresh', native_stage_local=dict(block, **(before or {}))),
            dict(label='after-fresh', native_stage_local=dict(block, chunks=3, **(after or {})))])
        return box.selection_reasons(record, True), record

    def test_selection_receipts(self):
        reasons, record = self.receipts()
        self.assertEqual((reasons, record['stage_delta']['chunks']), ([], 3))
        self.assertIn('stage-selection-not-this-arm', self.receipts(before=dict(selected=False))[0])
        self.assertIn('stage-selection-not-this-arm', self.receipts(after=dict(selected=False))[0])
        self.assertIn('stage-selection-moved-during-the-fresh-prompt', self.receipts(after=dict(selections=2))[0])
        order = ['status', 'pause', 'native_stage_local', 'native_progress', 'release_auto_arm', 'resume']
        self.assertIn('control-sequence-not-pause-progress-selection-release-resume', self.receipts(history=order)[0])
        twice = ARM_ACTIONS[:4] + ['native_stage_local'] + ARM_ACTIONS[4:]
        self.assertIn('arm-sent-the-selection-more-than-once', self.receipts(history=twice)[0])
        self.assertEqual(self.receipts(after=dict(selected=None))[0], ['stage-selection-not-this-arm'])

    def test_stage_view_needs_the_block(self):
        self.assertIsNone(box.stage_view({}))
        self.assertIsNone(box.stage_view(dict(native_stage_local=dict(selected=True))))
        row = box.stage_view(dict(native_stage_local=dict(selected=False, ring_reserve=3, poll_reserve=5,
                                                          counters=dict.fromkeys(box.STAGE_COUNTERS, 2))))
        self.assertEqual((row['selected'], row['chunks'], row['ring_reserve'], row['poll_reserve']), (False, 2, 3, 5))


class Digests(unittest.TestCase):
    def test_stream_digests_use_the_first_tokens_only(self):
        tokens = list(range(1000))
        states = [SimpleNamespace(index=0, tokens=tokens), SimpleNamespace(index=1, tokens=tokens[:box.STREAM_PREFIX_TOKENS]),
                  SimpleNamespace(index=2, tokens=tokens[:9])]
        rows = box.stream_digests(states)
        self.assertEqual(rows['0']['prefix_sha256'], rows['1']['prefix_sha256'])
        self.assertEqual((rows['0']['tokens'], rows['0']['prefix_tokens']), (1000, box.STREAM_PREFIX_TOKENS))
        self.assertEqual(rows['2']['prefix_tokens'], 9)
        self.assertNotEqual(rows['2']['prefix_sha256'], rows['0']['prefix_sha256'])


# --- grade --------------------------------------------------------------------------------------------

GPU = {'local': (22000, 23000, 400), 'serial': (22000, 23000, 401)}
SEGMENTS = (('prepare', 0), ('prefix', 0), ('layer', 0), ('suffix', 0), ('pp_send', 0), ('state_finalize', 0),
            ('prepare', 1), ('pp_receive', 1), ('prefix', 1), ('layer', 1), ('suffix', 1), ('state_finalize', 1))


def outputs(selection, index, *, fresh=(11,), append=(22,), prefix=box.STREAM_PREFIX_TOKENS):
    return dict(fresh=list(fresh), append=list(append),
                standing={str(i): dict(tokens=900, prefix_tokens=prefix, prefix_sha256='%064d' % (i + 1)) for i in range(8)})


class StageEmit(hf.Emit):
    """hf.Emit over the two paths' command rows (the real row layout, real emitters).

    Serial: every command is answered by all four ranks, and one release. Local: an open and a release per stage answered by
    that stage's ranks, each non-target rank's row written with replied 0 as executor_hooks does, the sender's receipt last.
    """

    def stage_command(self, kind, segment, stage, ranks, origin, direct):
        self.seq += 1
        submit = self.now
        sent = submit + self.SEND * 1000
        recv = {}
        method = {'release': 'flash_s5_release', 'open': 'flash_s5_open', 'receipt': 'flash_s5_receipt'}.get(kind, 'flash_s5_submit')
        for rank, trace in self.workers.items():
            enter = sent + self.FIFO * 1000
            if kind == 'release':
                for text in trace.lines():
                    self.log(enter, rank, text)
            done = enter + self.HOST * 1000 if rank in ranks else enter + 1000
            emit = done + (self.HOL - (3 - rank)) * 1000
            if rank in ranks:
                recv[rank] = emit + self.MQ * 1000
                trace.record('r', self.seq, method, enter, done, emit, direct, 1)
            else:
                trace.record('r', self.seq, method, enter, done, None, 0, 0)
        apply = max(recv.values()) + self.APPLY * 1000
        self.engine.record('n', self.seq, kind, segment, self.seq if kind == 'segment' else None, stage,
                           0 if segment == 'layer' else None, origin, submit, sent,
                           recv.get(0), recv.get(1), recv.get(2), recv.get(3), apply, 'ok', direct)
        self.now = apply

    def chunk(self, start, end, *, selection, live):
        self.txn += 1
        local = selection == 'local'
        first = self.now
        all_ranks = (0, 1, 2, 3)
        stage_ranks = {0: (0, 1), 1: (2, 3)}
        for stage in (0, 1) if local else (None,):
            self.stage_command('open', None, stage, stage_ranks[stage] if local else all_ranks, 'tick', 0)
            self.tick(self.CADENCE)
        for i, (segment, stage) in enumerate(SEGMENTS):
            self.stage_command('segment', segment, stage, stage_ranks[stage] if local else all_ranks,
                               'wait' if live and i <= 1 else 'tick', 1)
            self.tick(self.CADENCE)
            if i == 1 and live:
                self.stock('execute_model')
                self.stock('sample_tokens')
                self.engine.record('w', self.seq, 'sample_tokens', self.now, self.now + 300_000, 2, 0)
        if local:
            for stage in (0, 1):
                self.stage_command('release', None, stage, stage_ranks[stage], 'tick', 0)
                self.tick(self.CADENCE)
            self.stage_command('receipt', None, None, stage_ranks[0], 'tick', 0)
            self.log(self.now, 'engine', 'flash-s5 stage-local txn=%d chunk=%d-%d commands=9,9 both_stages_ns=2000000 first_ns=%d,%d '
                                         'last_ns=%d,%d local_release_ns=%d,%d headroom_refused=1' % (
                                             self.txn, start, end, first, first, self.now, self.now, self.now - 5000, self.now))
        else:
            self.stage_command('release', None, None, all_ranks, 'tick', 0)
        for text in self.engine.lines(txn=self.txn, request_generation=1, chunk_start=start, chunk_end=end,
                                      decode_live=live, mode='burst', open_rows=0):
            self.log(self.tick(100), 'engine', text)

    def memory(self, selection, rank, txn, start, end):
        allocated, reserved, free = GPU[selection]
        for point, retries, shift in (('open', 5 + txn, 0), ('released', 6 + txn, 1)):
            self.log(self.now, rank, 'flash-s5 memory point=%s rank=%d txn_seq=%d chunk=%d-%d allocated=%d reserved=%d '
                                     'inactive_split=%d alloc_retries=%d ooms=0 free=%d total=25296044032' % (
                                         point, rank, txn, start, end, (allocated + shift) * 2 ** 20, reserved * 2 ** 20,
                                         2 ** 26, retries, (free - shift) * 2 ** 20))


def synthetic_run(order=None, extra=None, *, rate=None):
    """The declared arms in order, each arm's real emitter lines, memory receipts and stage counters."""
    emit, arms, extra = StageEmit(), [], extra or {}
    for selection, cohort, index in (order or box.ARM_ORDER):
        name, band = box.label(selection, cohort, index), ap.band_of_index(index)
        cuts = tap.band_cuts(band)
        start = emit.tick(5_000_000)
        emit.tick(500_000)
        chunks = len(cuts) - 1
        lives = [8] * chunks if cohort == 'live' else [8] + [0] * (chunks - 1)
        for c, live in enumerate(lives):
            for rank in range(4):
                emit.memory(selection, rank, c + 1, cuts[c], cuts[c + 1])
            emit.chunk(cuts[c], cuts[c + 1], selection=selection, live=live)
        if name in extra:
            emit.log(emit.now, 1, extra[name])
        end = emit.tick(1_000_000)
        streams = [40.0, 41.0, 42.0, 42.0, 43.0, 43.0, 44.0, 45.0]
        window = {'aggregate_tokens_per_s': 340.0, 'stream_rates_tokens_per_s': streams, 'gap_ms_p99': 60.0, 'gap_ms_max': 100.0}
        client = {'prefill_tokens_per_s': (rate or {}).get(selection, 900.0), 'all_open_share_of_span': 1.0 if cohort == 'live' else 0.1}
        if cohort == 'live':
            client['decode_window'] = window
        local = selection == 'local'
        delta = dict(selections=0, chunks=chunks if local else 0, headroom_refused=chunks if local else 0,
                     local_releases=2 * chunks if local else 0, both_stages_ns=2000000 * chunks if local else 0)
        arms.append(dict(label=name, selection=selection, mode='burst', cohort=cohort, index=index, band=band, complete=True,
                         reasons=[], wall_start_s=emit.wall(start), wall_end_s=emit.wall(end),
                         progress_delta={'chunks': chunks, 'commands': chunks * 16}, stage_delta=delta,
                         selection_switch=dict(local=local, route='full'), outputs=outputs(selection, index),
                         fresh={'prompt_tokens_observed': cuts[-1] + 1, 'ttft_s': float(chunks)},
                         decode_only={'aggregate_tokens_per_s': 400.0, 'stream_rates_tokens_per_s': [50.0] * 8},
                         client_basis=client, append={'ttft_s': 2.0}, metrics={}))
    return emit, arms


class Grade(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def grade(self, emit, arms, ingress=None, watches=None, lines=None):
        q = Path(tempfile.mkdtemp(dir=self.tmp.name))
        tap.write_streams(q, emit.lines if lines is None else lines)
        ingress = {a['label']: tap.PASS for a in arms} if ingress is None else ingress
        watches = {a['label']: tap.ALIVE for a in arms} if watches is None else watches
        return box.grade(arms, ingress, tap.CHECKS, [str(q / 'engine-stream.log'), str(q / 'engine-stream.err')], watches)

    def edited(self, arms, name, **fields):
        return [dict(a, **fields) if a['label'] == name else a for a in arms]

    def test_eight_arms_pass_with_all_sections(self):
        emit, arms = synthetic_run()
        result = self.grade(emit, arms)
        self.assertEqual((result['verdict'], result['missing'], result['failures'], result['classification']),
                         ('PASS', [], [], None), result['missing'])
        self.assertEqual(result['required_arms'], LABELS)
        self.assertEqual(len(LABELS), 8)
        self.assertEqual({v['verdict'] for v in result['validity'].values()}, {'PASS'})
        row = result['table']['local-live-0']
        self.assertEqual((row['band'], row['selection'], row['mode'], row['cohort']), (1, 'local', 'burst', 'live'))
        self.assertEqual((row['min_stream_rate'], row['gap_p99_ms'], row['gap_max_ms']), (40.0, 60.0, 100.0))
        self.assertAlmostEqual(row['p10_stream_rate'], 40.7)
        self.assertAlmostEqual(row['p10_stream_ratio'], 40.7 / 50)
        self.assertEqual(len(row['stream_ratios']), 8)
        self.assertEqual((row['stage']['commands_per_chunk'], row['stage']['both_stages_s_per_chunk']), ([9, 9], 0.002))
        self.assertEqual(result['table']['serial-live-0']['stage'], box.stage_summary([]))
        engaged = result['validity']['local-live-0']['engagement']
        self.assertEqual((engaged['engaged'], engaged['foreign_replies']), (True, 0))
        self.assertEqual(engaged['stage_lines'], engaged['owned_chunks'])
        engaged = result['validity']['serial-live-0']['engagement']
        self.assertEqual((engaged['engaged'], engaged['missing_replies'], engaged['stage_lines']), (True, 0, 0))
        summary = result['summary']
        self.assertEqual(sorted(summary['groups']), ['local-live', 'local-solo', 'serial-live', 'serial-solo'])
        self.assertEqual(summary['groups']['local-live']['arms'], ['local-live-0', 'local-live-1'])
        self.assertEqual(summary['groups']['local-live']['both_stages_s_per_chunk'], 0.002)
        self.assertEqual(summary['local_over_serial']['live']['prefill_tok_s'], 1.0)
        pair = result['memory_local_vs_serial']['live-0']
        self.assertEqual((pair['local_minus_serial']['peak_reserved_mib'], pair['local_minus_serial']['min_free_mib']), (0.0, -1.0))
        self.assertEqual((pair['zero_ooms'], pair['min_free_within_margin']), (True, True))
        self.assertEqual(sorted(result['memory_local_vs_serial']), ['live-0', 'live-1', 'solo-0', 'solo-1'])
        self.assertEqual((result['speed_is_gate'], result['memory_is_gate']), (False, False))
        self.assertEqual(result['numerics']['verdict'], 'IDENTICAL')
        self.assertIn('Not assessed by this queue', result['release_readiness'])

    def test_speed_and_memory_never_change_the_verdict(self):
        emit, arms = synthetic_run(rate=dict(local=1.0))
        self.assertEqual(self.grade(emit, arms)['verdict'], 'PASS')

    def test_a_local_arm_that_did_not_engage_is_inc(self):
        emit, arms = synthetic_run()
        base_delta = next(a['stage_delta'] for a in arms if a['label'] == 'local-live-0')
        for field, value, reason in (('chunks', 0, 'stage-chunks-delta-differs-from-the-auto-owned-chunks'),
                                     ('local_releases', 1, 'stage-releases-not-two-per-chunk')):
            with self.subTest(field=field, value=value):
                result = self.grade(emit, self.edited(arms, 'local-live-0', stage_delta=dict(base_delta, **{field: value})))
                self.assertEqual((result['verdict'], result['failures']), ('INC', []))
                self.assertIn('local-live-0:not-engaged', result['missing'])
                row = result['validity']['local-live-0']
                self.assertEqual((row['verdict'], row['classification'], row['reasons']), ('INC', None, [reason]))
        result = self.grade(emit, self.edited(arms, 'local-live-0', stage_delta=None))
        self.assertEqual(result['validity']['local-live-0']['reasons'], ['engagement-receipts-missing'])
        self.assertEqual(result['verdict'], 'INC')

    def test_lines_of_the_other_path_under_this_arms_receipts_are_inc(self):
        order = [(('serial' if s == 'local' else 'local'), c, i) for s, c, i in box.ARM_ORDER]
        emit, arms = synthetic_run()
        other_emit, other = synthetic_run(order)
        relabel = {box.label(*old): box.label(*new) for old, new in zip(order, box.ARM_ORDER)}
        swapped = [dict(a, label=relabel[a['label']], selection=next(x for x in arms if x['label'] == relabel[a['label']])['selection'],
                        stage_delta=next(x for x in arms if x['label'] == relabel[a['label']])['stage_delta']) for a in other]
        result = self.grade(other_emit, swapped)
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('non-target-rank-replied-to-a-stage-command', result['validity']['local-solo-0']['reasons'])
        self.assertIn('stage-lines-differ-from-the-auto-owned-chunks', result['validity']['local-solo-0']['reasons'])
        self.assertIn('serial-arm-logged-a-stage-line', result['validity']['serial-solo-0']['reasons'])
        self.assertIn('serial-arm-stage-command-without-four-replies', result['validity']['serial-solo-0']['reasons'])

    def test_a_serial_arm_that_moved_a_stage_counter_is_inc(self):
        emit, arms = synthetic_run()
        for field in box.STAGE_COUNTERS:
            with self.subTest(field=field):
                delta = dict.fromkeys(box.STAGE_COUNTERS, 0)
                delta[field] = 1
                result = self.grade(emit, self.edited(arms, 'serial-solo-1', stage_delta=delta))
                self.assertEqual(result['validity']['serial-solo-1']['reasons'], ['serial-arm-moved-a-stage-counter'])
                self.assertEqual(result['verdict'], 'INC')

    def test_each_engine_signature_fails_that_arm_as_engine(self):
        for line, name in [('(EngineCore_DP0 pid=101) ERROR native driver failed: %s' % text, name) for name, text in box.SIGNATURES] + [
                ('(Worker_TP1 pid=202) ERROR QSA layers disagree on the packed selection width', 'tp-disagree')]:
            with self.subTest(name=name):
                emit, arms = synthetic_run(extra={'local-live-1': line})
                result = self.grade(emit, arms)
                row = result['validity']['local-live-1']
                self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))
                self.assertEqual((row['verdict'], row['classification']), ('FAIL', 'ENGINE'))
                self.assertIn(name, row['reasons'])
                self.assertEqual({v['verdict'] for k, v in result['validity'].items() if k != 'local-live-1'}, {'PASS'})
                self.assertEqual(result['summary']['arm_verdicts']['local-live-1'], 'FAIL')

    def test_a_signature_between_the_arm_spans_fails_the_run(self):
        emit, arms = synthetic_run()
        gap = arms[3]['wall_end_s'] + progress.LOG_LAG_S + 0.5
        emit.log(emit.mono0 + int(round((gap - hf.WALL0) * 1e9)), 1, 'ERROR stage-local commit without release readiness')
        result = self.grade(emit, arms)
        self.assertEqual((result['verdict'], result['failures'], result['engine_lines_outside_every_arm']),
                         ('FAIL', ['engine-signature-outside-every-arm-span'], 1))
        self.assertEqual({v['verdict'] for v in result['validity'].values()}, {'PASS'})

    def test_the_stage_line_and_progress_rows_are_no_signature(self):
        self.assertIsNone(box.signature_of(progress.PROGRESS_LINE + '{"note":"stage-local commit without"}'))
        self.assertIsNone(box.signature_of('flash-s5 stage-local txn=1 chunk=0-1024 commands=1,1 both_stages_ns=0'))
        self.assertEqual(box.signature_of('x stage-local commit without y'), 'stage-commit')

    def test_the_error_of_an_arm_counts_its_signature(self):
        emit, arms = synthetic_run()
        changed = self.edited(arms, 'local-solo-0', complete=False, reasons=['arm-error'],
                              error={'error_message': 'native-response: a stage command targets another stage'})
        row = self.grade(emit, changed)['validity']['local-solo-0']
        self.assertEqual((row['verdict'], row['classification'], row['signatures']), ('FAIL', 'ENGINE', ['stage-target']))

    def test_engine_watch_ingress_and_incomplete_rows(self):
        emit, arms = synthetic_run()
        name = 'serial-live-0'
        watches = {a['label']: tap.ALIVE for a in arms}
        result = self.grade(emit, arms, watches=dict(watches, **{name: {'verdict': 'DEAD'}}))
        self.assertEqual((result['verdict'], result['classification'], result['validity'][name]['crash_free']),
                         ('FAIL', ['ENGINE'], False))
        kept = [a for a in arms if a['label'] != name]
        result = self.grade(emit, kept, watches=dict(watches, **{name: {'verdict': 'DEAD'}}))
        self.assertEqual(result['validity'][name]['reasons'], ['engine-dead'])
        self.assertEqual(result['validity'][name]['classification'], 'ENGINE')
        watches.pop(name)
        result = self.grade(emit, kept, watches=watches)
        self.assertEqual((result['verdict'], result['validity'][name]['verdict']), ('INC', 'INC'))
        self.assertIn('declared-arm-missing:' + name, result['missing'])
        result = self.grade(emit, arms, watches=watches)
        self.assertEqual((result['verdict'], result['validity'][name]['reasons']), ('INC', ['engine-watch-record-missing']))
        result = self.grade(emit, self.edited(arms, name, complete=False, reasons=['fresh-not-complete']))
        self.assertEqual((result['verdict'], result['validity'][name]['verdict']), ('INC', 'INC'))
        result = self.grade(emit, arms, ingress=dict({a['label']: tap.PASS for a in arms}, **{name: {'verdict': 'FAIL'}}))
        self.assertEqual((result['verdict'], result['classification'], result['validity'][name]['classification']),
                         ('FAIL', ['HARNESS'], 'HARNESS'))

    def test_a_receipt_row_rides_past_only_the_runs_last_flush(self):
        # A local arm is last: its last chunk's receipt row is on no flush, and the join accepts that one.
        order = list(box.ARM_ORDER)
        order[-1], order[-2] = order[-2], order[-1]
        emit, arms = synthetic_run(order)
        result = self.grade(emit, arms)
        self.assertNotIn('progress-local-solo-1:worker-rows-missing', result['missing'])
        chunks = result['progress']['arms']['local-solo-1']['chunks']
        self.assertEqual(chunks[-1]['late_receipt_rows'], 1)
        self.assertEqual(sum(c['late_receipt_rows'] for c in chunks[:-1]), 0)
        # A required row missing with no lost flush to explain it still fails the join: Tears holds the explained case.
        span = next(a for a in arms if a['label'] == 'local-solo-0')
        inside = [i for i, (s, t) in enumerate(emit.lines) if s == 'err' and progress.PROGRESS_LINE in t
                  and span['wall_start_s'] <= progress.docker_stamp(t) <= span['wall_end_s'] and '"side":"worker"' in t]
        lines = list(emit.lines)
        at = inside[0]
        body = json.loads(lines[at][1][lines[at][1].index(progress.PROGRESS_LINE) + len(progress.PROGRESS_LINE):])
        body['rows']['r'] = body['rows']['r'][1:]
        lines[at] = (lines[at][0], lines[at][1][:lines[at][1].index(progress.PROGRESS_LINE)] + progress.PROGRESS_LINE
                     + json.dumps(body, separators=(',', ':'), sort_keys=True) + '\n')
        result = self.grade(emit, arms, lines=lines)
        self.assertEqual(result['verdict'], 'INC', result['missing'])
        self.assertTrue(any(x.startswith('progress-') and x.endswith(':worker-rows-missing') for x in result['missing']), result['missing'])

    def test_held_f_completeness_defects_hold(self):
        emit, arms = synthetic_run()
        changed = self.edited(arms, 'local-live-1', band=2)
        self.assertIn('local-live-1:band-not-the-round-index-band', self.grade(emit, changed)['missing'])
        result = self.grade(emit, self.edited(arms, 'local-live-1', progress_delta={'chunks': 99}))
        self.assertIn('progress-local-live-1:progress-lines-differ-from-the-chunk-counter', result['missing'])
        result = self.grade(emit, arms, lines=[])
        self.assertIn('no-progress-lines-in-the-engine-log', result['missing'])
        self.assertEqual(result['verdict'], 'INC')

    def test_memory_receipts_read_the_recorded_line_and_the_counter_deltas(self):
        line = ('flash-s5 memory point=open rank=0 txn_seq=1 chunk=0-1024 allocated=22696461824 reserved=23337107456 '
                'inactive_split=124746240 alloc_retries=5 ooms=0 free=693829632 total=25296044032')
        later = line.replace('point=open', 'point=released').replace('alloc_retries=5', 'alloc_retries=9').replace(
            'ooms=0', 'ooms=1').replace('allocated=22696461824', 'allocated=22700000000').replace('free=693829632', 'free=600000000')
        row = box.memory_receipts([(None, line), (None, later)])['0']
        self.assertEqual(row, dict(receipts=2, peak_allocated_mib=21648.4, peak_reserved_mib=22256.0, min_free_mib=572.2,
                                   alloc_retries=4, ooms=1))
        self.assertEqual(box.memory_worst({}), None)

    def test_the_memory_gates_are_reported_and_change_no_verdict(self):
        emit, arms = synthetic_run()
        table = {name: dict(memory=dict(worst=dict(peak_allocated_mib=1.0, peak_reserved_mib=1.0, min_free_mib=f, alloc_retries=0,
                                                   ooms=o, receipts=1))) for name, f, o in (
            ('local-live-0', 7.9, 0), ('serial-live-0', 10.0, 0), ('local-solo-0', 8.0, 1), ('serial-solo-0', 10.0, 0))}
        rows = box.memory_compare(table)
        self.assertEqual((rows['live-0']['min_free_within_margin'], rows['live-0']['zero_ooms']), (False, True))
        self.assertEqual((rows['solo-0']['min_free_within_margin'], rows['solo-0']['zero_ooms']), (True, False))
        self.assertIsNone(rows['live-1'])

    def test_the_report_keeps_three_separate_sections(self):
        emit, arms = synthetic_run()
        result = self.grade(emit, arms)
        text = box.report(result)
        heads = re.findall(r'^## (.+)$', text, flags=re.M)
        self.assertEqual(heads, ['Mechanism and numerics', 'Performance', 'Release readiness'])
        self.assertEqual(text.split('## Release readiness\n\n')[1].strip(), box.RELEASE_READINESS)
        mechanism = text[text.index('## Mechanism'):text.index('## Performance')]
        for name in LABELS:
            self.assertIn('| %s | PASS |' % name, mechanism)
        self.assertIn('Numerics: **IDENTICAL**', mechanism)
        performance = text[text.index('## Performance'):text.index('## Release readiness')]
        for field in ('Local over serial', 'Device memory', 'TTFT', 'p10 stream ratio', 'both stages s/chunk', 'ring refusals'):
            self.assertIn(field, performance)
        self.assertNotIn('PASS', performance)
        kept = [a for a in arms if a['label'] != 'local-solo-1']
        text = box.report(self.grade(emit, kept))
        self.assertIn('Overall: **INC**', text)
        self.assertIn('| solo-1 | not available |', text)

    def test_grade_cli_files_rcs_and_no_overwrite(self):
        emit, arms = synthetic_run()
        q = Path(self.tmp.name)
        tap.write_streams(q, emit.lines)
        args = []
        for a in arms:
            for prefix, data, flag in (('arm', a, None), ('ingress', tap.PASS, '--ingress'), ('watch', tap.ALIVE, '--watch')):
                path = q / ('%s-%s.json' % (prefix, a['label']))
                path.write_text(json.dumps(data))
                args += ['--arm', str(path)] if flag is None else [flag, '%s=%s' % (a['label'], path)]
        for name in tap.CHECKS:
            (q / (name + '.json')).write_text(json.dumps(tap.PASS))
            args += ['--check', '%s=%s' % (name, q / (name + '.json'))]
        args += ['--log', str(q / 'engine-stream.log'), '--log', str(q / 'engine-stream.err'), '--report', str(q / 'r.md')]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(box.main(['grade'] + args + ['--out', str(q / 'grade.json')]), 0)
        self.assertEqual(out.getvalue().strip(), 'ALPHA_C0_GRADE PASS')
        self.assertEqual(json.loads((q / 'grade.json').read_text())['schema'], box.GRADE_SCHEMA)
        self.assertTrue((q / 'r.md').read_text().startswith('# q18a-alpha-c0 report'))
        self.assertEqual(box.main(['grade'] + args + ['--out', str(q / 'grade.json')]), 2)
        (q / 'arm-local-solo-1.json').write_text('{not json')
        args[args.index(str(q / 'r.md'))] = str(q / 'r2.md')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(box.main(['grade'] + args + ['--out', str(q / 'grade-2.json')]), 2)
        self.assertIn('declared-arm-missing:local-solo-1', json.loads((q / 'grade-2.json').read_text())['missing'])


class Tears(unittest.TestCase):
    """The pre-registered rule, over the real emitters' lines damaged the way q18a-alpha-perf's box log was.

    The two shapes are that log's lines 3832 with 3835 (bytes moved to a later line, a foreign line in their place) and 34985
    (a header landing inside the JSON).
    """
    FOREIGN = '(Worker_PP1_TP0_EP0 pid=124) INFO 11-01 20:06:28 [qsa_operation.py:797] flash-s5 native-alloc rank=2 txn_seq=64 chunk=1-2'

    def setUp(self):
        self.helper = Grade(methodName='grade')
        self.helper.setUp()
        self.addCleanup(self.helper.tmp.cleanup)
        # Enough chunks to lose some: the band's prompt in sixteen pieces, against band_cuts' few.
        real = tap.band_cuts

        def cuts(band):
            ends = real(band)
            step = (ends[-1] - ends[0]) // 16
            return [ends[0] + k * step for k in range(16)] + [ends[-1]]
        with mock.patch.object(tap, 'band_cuts', cuts):
            self.emit, self.arms = synthetic_run()

    def grade(self, lines=None, arms=None):
        return self.helper.grade(self.emit, arms or self.arms, lines=self.emit.lines if lines is None else lines)

    def span(self, name):
        return next(a for a in self.arms if a['label'] == name)

    def inside(self, name, mark):
        span = self.span(name)
        return [i for i, (_, t) in enumerate(self.emit.lines) if mark in t
                and span['wall_start_s'] <= progress.docker_stamp(t) <= span['wall_end_s']]

    @staticmethod
    def moved(line, foreign):
        """3832's shape: the body stops inside a row, a foreign line follows; the rest is returned as a line of its own."""
        stream, text = line
        at = text.index('"rows":')
        cut = at + min(200, (len(text) - at) // 2)
        stamp = text.split(' ', 1)[0]
        return (stream, text[:cut] + foreign + '\n'), (stream, stamp + ' ' + text[cut:])

    @staticmethod
    def spliced(line):
        """34985's shape: a header inside a row; nothing else moved."""
        stream, text = line
        at = text.index('"rows":')
        cut = at + min(200, (len(text) - at) // 2)
        return stream, text[:cut] + '(Worker_PP0_TP1_EP1 pid=102) ' + text[cut:]

    def tear(self, indexes, shape='spliced'):
        lines = list(self.emit.lines)
        for i in sorted(indexes, reverse=True):
            if shape == 'spliced':
                lines[i] = self.spliced(lines[i])
            else:
                lines[i:i + 1] = self.moved(lines[i], self.FOREIGN)
        return lines

    def worker_lines(self, name, rank=None):
        mark = '"rank":%d,' % rank if rank is not None else '"side":"worker"'
        return [i for i in self.inside(name, progress.PROGRESS_LINE) if mark in self.emit.lines[i][1] and '"side":"worker"' in self.emit.lines[i][1]]

    def test_a_clean_run_counts_nothing_and_the_recount_equals_the_joins_medians(self):
        result = self.grade()
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        tears = result['progress']['tears']
        self.assertEqual((tears['torn_progress_lines'], tears['lost_flushes'], tears['lost_flushes_outside_every_arm']), (0, 0, []))
        for name, row in result['progress']['arms'].items():
            self.assertEqual(row['tears']['lost_records'], 0)
            self.assertTrue(row['tears']['within_limits'])
            self.assertFalse(any('_seqs' in c for c in row['chunks']))
            before = json.dumps(dict(bands=row['bands'], all=row['all_counted'], n=row['counted_chunks']), sort_keys=True)
            box.recount(row)
            self.assertEqual(json.dumps(dict(bands=row['bands'], all=row['all_counted'], n=row['counted_chunks']), sort_keys=True), before)

    def test_one_torn_worker_line_of_either_real_shape_is_counted_and_its_chunks_drop_out(self):
        clean = self.grade()['progress']['arms']['local-solo-0']
        for shape in ('spliced', 'moved'):
            i = self.worker_lines('local-solo-0', 3)[2]
            result = self.grade(self.tear([i], shape))
            self.assertEqual(result['verdict'], 'PASS', (shape, result['missing']))
            row = result['progress']['arms']['local-solo-0']
            tears = row['tears']
            self.assertEqual((tears['torn_progress_lines'], tears['lost_records'], tears['within_limits']), (1, 1, True), shape)
            self.assertEqual([x['rank'] for x in tears['lost_flushes']], [3])
            self.assertTrue(1 <= len(tears['dropped_chunk_starts']) <= 2, tears)
            self.assertEqual(row['reasons'], [])
            gone = [c for c in row['chunks'] if c.get('dropped_out') and c['label'] == row['counted_label']]
            self.assertEqual(row['counted_chunks'], clean['counted_chunks'] - len(gone))
            self.assertEqual(sum(b['chunks'] for b in row['bands'].values()), row['counted_chunks'])
            self.assertEqual({c['chunk_start'] for c in row['chunks'] if c.get('dropped_out')}, set(tears['dropped_chunk_starts']))
            self.assertEqual(len(row['chunks']), len(clean['chunks']), 'a dropped chunk stays listed')
            self.assertEqual(result['validity']['local-solo-0']['verdict'], 'PASS')
            self.assertEqual(result['progress']['arms']['serial-solo-0']['tears']['lost_records'], 0)
            self.assertIn('### Torn lines and lost records', box.report(result))

    def test_the_same_tear_on_a_serial_arm(self):
        result = self.grade(self.tear([self.worker_lines('serial-live-1', 2)[1]]))
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        self.assertEqual(result['progress']['arms']['serial-live-1']['tears']['lost_records'], 1)

    def test_two_lost_flushes_in_one_arm_go_inc_and_one_in_each_of_two_arms_does_not(self):
        a, b = self.worker_lines('local-live-0', 1)[1], self.worker_lines('local-live-0', 3)[5]
        result = self.grade(self.tear([a, b]))
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('progress-local-live-0:lost-records-above-the-declared-limit:2>1', result['missing'])
        self.assertEqual(result['validity']['local-live-0']['verdict'], 'INC')
        self.assertEqual(result['validity']['local-solo-0']['verdict'], 'PASS')
        self.assertFalse(result['progress']['arms']['local-live-0']['tears']['within_limits'])
        result = self.grade(self.tear([a, self.worker_lines('serial-live-0', 3)[5]]))
        self.assertEqual(result['verdict'], 'PASS', result['missing'])

    def test_a_flush_gone_with_no_torn_text_is_found_by_its_writers_line_gap(self):
        # A tear can destroy the marker itself: nothing unparsable is left, the gap still shows.
        i = self.worker_lines('local-solo-1', 0)[2]
        result = self.grade([x for k, x in enumerate(self.emit.lines) if k != i])
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        tears = result['progress']['arms']['local-solo-1']['tears']
        self.assertEqual((tears['torn_progress_lines'], tears['lost_records']), (0, 1))
        self.assertEqual(tears['lost_flushes'][0]['evidence'], ['line-gap'])
        self.assertTrue(tears['dropped_chunk_starts'])

    def test_a_lost_engine_flush_drops_its_chunk_and_clears_only_what_it_explains(self):
        engine = [i for i in self.inside('local-live-1', progress.PROGRESS_LINE) if '"side":"engine"' in self.emit.lines[i][1]]
        result = self.grade(self.tear([engine[3]]))
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        row = result['progress']['arms']['local-live-1']
        self.assertEqual((row['tears']['lost_records'], len(row['tears']['dropped_chunk_starts'])), (1, 1))
        self.assertEqual(row['reasons'], [])
        self.assertEqual(row['counted_chunks'], len(row['chunks']))
        hole = row['tears']['dropped_chunk_starts'][0]
        self.assertNotIn(hole, [c['chunk_start'] for c in row['chunks']])
        before = next(c for c in row['chunks'] if c['chunk_end'] == hole)
        self.assertNotIn('open_interval_s', before)
        # The same missing line with a counter that also disagrees stays INC: the shortfall is not the loss.
        arms = [dict(a, progress_delta=dict(a['progress_delta'], chunks=a['progress_delta']['chunks'] + 1)) if a['label'] == 'local-live-1' else a
                for a in self.arms]
        result = self.grade(self.tear([engine[3]]), arms)
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('progress-local-live-1:progress-lines-differ-from-the-chunk-counter', result['missing'])

    def test_the_first_and_the_last_engine_flush_of_a_solo_arm(self):
        for pick in (0, -1):
            engine = [i for i in self.inside('serial-solo-1', progress.PROGRESS_LINE) if '"side":"engine"' in self.emit.lines[i][1]]
            result = self.grade(self.tear([engine[pick]]))
            row = result['progress']['arms']['serial-solo-1']
            self.assertEqual(row['reasons'], [], pick)
            self.assertEqual(row['tears']['lost_records'], 1)

    def test_a_torn_stage_line_is_treated_the_same(self):
        i = self.inside('local-live-0', box.STAGE_LINE)[4]
        lines = list(self.emit.lines)
        stream, text = lines[i]
        cut = text.index('both_stages_ns=') + 17
        lines[i] = (stream, text[:cut] + '(Worker_PP0_TP1_EP1 pid=102) ' + text[cut:])
        result = self.grade(lines)
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        tears = result['validity']['local-live-0']['tears']
        self.assertEqual((tears['torn_stage_lines'], tears['lost_records'], len(tears['dropped_chunk_starts'])), (1, 1, 1))
        engaged = result['validity']['local-live-0']['engagement']
        self.assertEqual((engaged['engaged'], engaged['lost_stage_lines']), (True, 1))
        table = result['table']['local-live-0']['stage']
        self.assertEqual(table['lines'], len(self.inside('local-live-0', box.STAGE_LINE)) - 1)
        row = result['progress']['arms']['local-live-0']
        self.assertEqual(row['counted_chunks'], len(row['chunks']) - 1)

    def test_a_torn_stage_line_on_a_serial_arm_is_still_a_stage_line(self):
        emit, arms = synthetic_run(extra={'serial-solo-0': 'flash-s5 stage-local txn=1 chunk=(Worker_PP0_TP0_EP0 pid=96) 0-1024'})
        result = self.helper.grade(emit, arms)
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('serial-arm-logged-a-stage-line', result['validity']['serial-solo-0']['reasons'])

    def test_two_missing_stage_lines_with_one_torn_stay_unexplained(self):
        at = self.inside('local-solo-1', box.STAGE_LINE)
        lines = list(self.emit.lines)
        stream, text = lines[at[1]]
        lines[at[1]] = (stream, text.replace('commands=', 'comm(Worker_PP0_TP0_EP0 pid=96) ands='))
        del lines[at[4]]
        result = self.grade(lines)
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('stage-lines-differ-from-the-auto-owned-chunks', result['validity']['local-solo-1']['reasons'])

    def test_a_torn_memory_line_is_counted_against_the_limit(self):
        at = self.inside('local-solo-0', 'flash-s5 memory ')
        lines = list(self.emit.lines)
        for i in at[:1]:
            lines[i] = (lines[i][0], lines[i][1].replace('allocated=', 'alloc(Worker_PP0_TP0_EP0 pid=96) ated='))
        result = self.grade(lines)
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        self.assertEqual(result['validity']['local-solo-0']['tears']['torn_memory_lines'], 1)
        self.assertEqual(result['validity']['local-solo-0']['tears']['dropped_chunk_starts'], [])
        lines[at[1]] = (lines[at[1]][0], lines[at[1]][1].replace('allocated=', 'alloc(Worker_PP0_TP0_EP0 pid=96) ated='))
        result = self.grade(lines)
        self.assertEqual(result['validity']['local-solo-0']['verdict'], 'INC')

    def test_a_header_inside_a_json_string_still_parses_and_is_still_a_tear(self):
        # The line reads and carries a mangled row kind: a parse check alone would feed it to the join.
        i = next(k for k in self.worker_lines('local-solo-0', 3)[2:] if '"flash_s5_' in self.emit.lines[k][1])
        lines = list(self.emit.lines)
        stream, text = lines[i]
        lines[i] = (stream, text.replace('"flash_s5_', '"flash_(Worker_PP0_TP1_EP1 pid=102) s5_', 1))
        at = lines[i][1].index(progress.PROGRESS_LINE) + len(progress.PROGRESS_LINE)
        json.loads(lines[i][1][at:])
        result = self.grade(lines)
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        tears = result['validity']['local-solo-0']['tears']
        self.assertEqual((tears['torn_progress_lines'], tears['lost_records'], [x['rank'] for x in tears['lost_flushes']]), (1, 1, [3]))
        self.assertTrue(tears['dropped_chunk_starts'])
        self.assertLess(result['progress']['worker_rows'], self.grade()['progress']['worker_rows'])
        # In the schema word the join lists it as off-schema; it is the same tear, and clears from that list only.
        lines[i] = (stream, text.replace('"schema":"s5-native', '"schema":"s5-(Worker_PP0_TP1_EP1 pid=102) native', 1))
        result = self.grade(lines)
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        self.assertEqual(result['validity']['local-solo-0']['tears']['torn_progress_lines'], 1)

    def test_a_line_off_schema_with_no_header_stays_a_problem(self):
        i = self.worker_lines('local-solo-0', 3)[2]
        lines = list(self.emit.lines)
        lines[i] = (lines[i][0], lines[i][1].replace('"schema":"s5-native', '"schema":"other-native', 1))
        self.assertIn('progress:progress-line-schema', self.grade(lines)['missing'])

    def test_a_header_alone_is_not_a_tear(self):
        # 525 such lines in q18a-alpha-perf, 2 tears: a doubled or absent header leaves the body whole.
        lines = [(s, t.replace(' INFO ', ' (Worker_PP0_TP0_EP0 pid=96) INFO ', 1)) for s, t in self.emit.lines]
        result = self.grade(lines)
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        self.assertEqual(result['progress']['tears']['torn_progress_lines'], 0)

    def test_a_signature_inside_a_torn_line_still_fails_the_arm(self):
        i = self.worker_lines('local-solo-0', 3)[2]
        for foreign, name in ((' ERROR stage-local commit without a release', 'stage-commit'),
                              (' ERROR tp disagree on rank 3', 'tp-disagree-in-a-torn-line')):
            lines = list(self.emit.lines)
            lines[i:i + 1] = self.moved(lines[i], self.FOREIGN[:29] + foreign)
            result = self.grade(lines)
            self.assertEqual(result['verdict'], 'FAIL', foreign)
            self.assertEqual(result['validity']['local-solo-0']['classification'], 'ENGINE')
            self.assertIn(name, result['validity']['local-solo-0']['signatures'])
        split = 'stage-local comm(Worker_PP0_TP0_EP0 pid=96) it without'
        self.assertEqual(box.signature_of('x ' + split), 'stage-commit')
        self.assertIsNone(box.signature_of('x ' + progress.PROGRESS_LINE + '{"schema":"s","note":"stage-local commit without"}'))

    def test_a_lost_flush_between_the_arms_is_reported_at_run_level_only(self):
        # A flush outside every span cannot touch a chunk: no arm pays for it.
        ledger = box.flush_ledger([], [(1.0, 'x ' + progress.PROGRESS_LINE + '{"line":7,"rank":2,"rows":[')])
        self.assertEqual(ledger['lost'][0]['line'], 7)
        arm = dict(label='a', wall_start_s=5.0, wall_end_s=6.0)
        row = dict(reasons=[], chunks=[])
        tears = box.settle_arm(arm, row, ledger, dict(progress=[], stage=[], memory=[]), [], None)
        self.assertEqual(tears['lost_records'], 0)

    def test_an_unreadable_identity_counts_as_its_own_loss(self):
        ledger = box.flush_ledger([], [(5.5, 'x ' + progress.PROGRESS_LINE + '{"base_ns":12(Worker_PP0_TP0_EP0 pid=96) 3,')])
        self.assertEqual((ledger['lost'], ledger['unidentified_torn_stamps']), ([], [5.5]))
        arm = dict(label='a', wall_start_s=5.0, wall_end_s=6.0)
        row = dict(reasons=[], chunks=[])
        self.assertEqual(box.settle_arm(arm, row, ledger, dict(progress=[], stage=[], memory=[]), [], None)['lost_records'], 1)

    def test_the_report_tool_drops_the_chunks_the_grade_drops(self):
        import alpha_c0_report as report
        i = self.worker_lines('local-solo-0', 3)[2]
        lines = self.tear([i], 'moved')
        result = self.grade(lines)
        root = Path(tempfile.mkdtemp(dir=self.helper.tmp.name))
        (root / 'engine-current.log').write_text(''.join(t for _, t in lines))
        for arm in self.arms:
            (root / ('arm-%s.json' % arm['label'])).write_text(json.dumps(arm))
        out = report.analyse(str(root))
        self.assertEqual(out['torn_progress_lines'], 1)
        self.assertEqual([x['rank'] for x in out['lost_flushes']], [3])
        self.assertEqual((out['errors'], out['problems']), ([], []))
        self.assertEqual(out['arms']['local-solo-0']['dropped_chunk_starts'], result['validity']['local-solo-0']['tears']['dropped_chunk_starts'])
        kept = {c['chunk_start'] for c in out['arms']['local-solo-0']['chunks']}
        self.assertFalse(kept & set(out['arms']['local-solo-0']['dropped_chunk_starts']))
        self.assertEqual(out['arms']['serial-solo-0']['dropped_chunk_starts'], [])

    def test_the_limits_are_the_declared_ones(self):
        self.assertEqual((box.LOST_RECORD_LIMIT, box.DROPPED_CHUNK_LIMIT), (1, 2))


class RealLines(unittest.TestCase):
    """The real stage driver, worker hooks and emitters on the per-rank ring (stage_net), their lines through the grade's own readers."""

    class Log:
        def __init__(self):
            self.lines = []

        def info(self, text, *args):
            self.lines.append(text % args if args else text)

    def run_stack(self, *, tokens=13):
        import collections
        import executor_hooks as eh
        import native_measurement as nm
        import stage_net
        import stage_progress as sp
        log, states = self.Log(), {rank: (nm.ProgressTrace('worker', rank), collections.deque()) for rank in range(4)}
        with mock.patch.object(eh, '_worker_progress', lambda rank: states[rank]), mock.patch.object(eh, '_progress_logger', lambda: log), \
                mock.patch.object(ed, '_progress_logger', lambda: log), mock.patch.object(sp, '_progress_logger', lambda: log):
            network = stage_net.RingStack(direct=True, tokens=tokens)
            self.addCleanup(network.close_network)
            driver = network.driver
            driver.auto_arm = SimpleNamespace(manual_owned=False, phase='native', active=False)
            driver._admission_intent = dict(origin='auto-arm', phase='native', seq=1)
            driver.install_progress('burst')
            driver.progress.mode = 'burst'
            network.run_request()
        return network, log.lines

    def records(self, lines):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = Path(directory) / 'engine-stream.err'
        path.write_text(''.join('2026-11-01T11:00:%02d.%06dZ x %s\n' % (i // 1000000, i % 1000000, text) for i, text in enumerate(lines)))
        return progress.read_lines([path])

    def test_a_real_stage_run_reads_through_the_join(self):
        network, lines = self.run_stack()
        chunks = network.expected_commits
        self.assertGreater(chunks, 1)
        records, errors = self.records(lines)
        self.assertEqual(errors, [])
        flushes, problems = progress.assemble(records)
        self.assertEqual(problems, [])
        workers = {}
        for flush in flushes:
            if flush['side'] == 'worker':
                for row in flush['rows'].get('r', []):
                    workers[(row['seq'], flush['rank'])] = row
        engine = sorted((f for f in flushes if f['side'] == 'engine'), key=lambda f: f['line'])
        self.assertEqual(len(engine), chunks)
        for index, chunk in enumerate(engine):
            info = progress.chunk_breakdown(chunk, workers)
            with self.subTest(chunk=index):
                self.assertEqual(info['foreign_replies'], 0)
                self.assertGreater(info['short_replies'], 0)
                self.assertEqual(info['kinds']['open'], 2)
                self.assertEqual(info['kinds']['release'], 2)
                if index < chunks - 1:
                    self.assertFalse(info['missing_worker_rows'], info['missing_worker_rows'])
                else:
                    # The run's last line: only the rows after the last flush may ride past it.
                    self.assertTrue({row['kind'] for row in info['missing_worker_rows']} <= {'release'}, info['missing_worker_rows'])
        rows = box.stage_lines([(None, text) for text in lines])
        self.assertEqual(len(rows), chunks)
        summary = box.stage_summary(rows)
        self.assertEqual(summary['lines'], chunks)
        self.assertEqual(len(summary['commands_per_chunk']), 2)
        self.assertGreater(min(summary['commands_per_chunk']), 0)
        # Each rank wrote a row for each command it read, a non-target's with replied 0 (the grade's foreign count reads these).
        skipped = [r for (seq, rank), r in workers.items() if r.get('replied') == 0]
        self.assertTrue(skipped)


class Numerics(unittest.TestCase):
    def arms(self, **changes):
        rows = []
        for selection, cohort, index in box.ARM_ORDER:
            name = box.label(selection, cohort, index)
            rows.append(dict(label=name, selection=selection, outputs=changes.get(name) or outputs(selection, index)))
        return rows

    def test_identical(self):
        result = box.numerics(self.arms())
        self.assertEqual((result['verdict'], result['differs'], result['unproven']), ('IDENTICAL', [], []))

    def test_a_local_arm_differs_while_the_serial_arms_agree(self):
        result = box.numerics(self.arms(**{'local-live-1': outputs('local', 1, fresh=(12,))}))
        self.assertEqual((result['verdict'], result['differs']), ('DIFFERS', ['fresh']))
        row = outputs('local', 1)
        row['standing']['3']['prefix_sha256'] = 'f' * 64
        result = box.numerics(self.arms(**{'local-live-1': row}))
        self.assertEqual((result['verdict'], result['differs']), ('DIFFERS', ['standing-3']))

    def test_the_serial_arms_already_differ_is_unproven_not_a_fault(self):
        result = box.numerics(self.arms(**{'serial-live-1': outputs('serial', 1, append=(23,))}))
        self.assertEqual((result['verdict'], result['differs']), ('UNPROVEN', []))
        self.assertIn('append-serial-arms-already-differ', result['unproven'])

    def test_missing_and_short_outputs_are_unproven(self):
        rows = [dict(a, outputs={}) for a in self.arms()]
        result = box.numerics(rows)
        self.assertEqual(result['verdict'], 'UNPROVEN')
        self.assertEqual(sorted(result['unproven']), ['append-tokens-missing', 'fresh-tokens-missing', 'standing-digests-missing'])
        short = {box.label(s, c, i): outputs(s, i, prefix=10) for s, c, i in box.ARM_ORDER if s == 'local'}
        result = box.numerics(self.arms(**short))
        self.assertEqual(result['verdict'], 'UNPROVEN')
        self.assertIn('standing-0-prefix-short', result['unproven'])

    def test_the_grade_fails_engine_on_differs_and_stays_inc_on_noise(self):
        grade = Grade('test_eight_arms_pass_with_all_sections')
        grade.setUp()
        emit, arms = synthetic_run()
        changed = grade.edited(arms, 'local-solo-1', outputs=outputs('local', 1, fresh=(99,)))
        result = grade.grade(emit, changed)
        self.assertEqual((result['verdict'], result['classification'], result['failures']),
                         ('FAIL', ['ENGINE'], ['numerics-differ:fresh']))
        changed = grade.edited(arms, 'serial-solo-1', outputs=outputs('serial', 1, fresh=(99,)))
        result = grade.grade(emit, changed)
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('numerics:fresh-serial-arms-already-differ', result['missing'])


# --- the derived arm loop under real bash ------------------------------------------------------------

STUB_BOX = tap.STUB_BOX.replace('e3_alpha_perf_box', 'e3_alpha_c0_box')


class ArmLoop(unittest.TestCase):
    """queue.BODY runs as written; arm and check clients are stubs (leaf arm-loop-clients), the watch and grade real."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='alpha-c0-loop-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q, self.h, self.bin, self.arms = (self.root / x for x in ('q', 'h', 'bin', 'arms'))
        for d in (self.q, self.h, self.bin, self.arms):
            d.mkdir()
        for name in ('native43_engine_death.py', 'alpha_c0_gate.py', 'native39_trunk_gate.py', 'native40_box_gate.py',
                     'native41_box_gate.py'):
            (self.h / name).symlink_to(HERE / name)
        (self.h / 'e3_native39_trunk_checks.py').write_text(tap.STUB_CHECKS)
        (self.h / 'e3_alpha_c0_box.py').write_text(STUB_BOX)
        self.tool('nvidia-smi', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')
        self.tool('docker', 'case "$*" in *State.Running*) echo true;; *) echo 2026-11-01T10:48:09.000000000Z;; esac')
        self.tool('systemctl', 'echo 0')
        (self.q / 'unarmed-stock-prefill.json').write_text(json.dumps(tap.PASS))

    tool = tap.ArmLoop.tool

    def run_loop(self, order, *, jump_after=None, die_in=None, timeout=240):
        emit, arms = synthetic_run(order)
        tap.write_streams(self.q, emit.lines)
        for a in arms:
            (self.arms / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
        draft = queue.derive()[DRAFT]
        self.assertIn(queue.BODY, draft)
        script = ['set -uo pipefail', 'FAILS=0; INCS=0; declare -A PHASE_VERDICT=()',
                  "ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=''; CONTAINER=qwen-flash-prod; SERVICE=vllm-flash; N0=0",
                  'Q=%s; H=%s; PROBE_URL=http://127.0.0.1:1; MODEL=m; RUN_ID=r; EPOCH=1; _cid=c; IDLE_TIMEOUT_S=1' % (self.q, self.h),
                  'Q18A_READINESS=/r/ready; S5_OVERLAY=/o; S5_LAUNCHER_NAME=l.sh', 'health_ok() { return 0; }']
        script += [tap.le_fix_function(draft, name) for name in tap.LOOP_FUNCTIONS]
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
                   ARMS=str(self.arms), REAL_HARNESS=str(HERE), DEATH_LINE=tap.DEATH_LINE, PYTHONDONTWRITEBYTECODE='1')
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
        return json.loads((self.q / 'alpha-c0-grade.json').read_text())

    def test_eight_arms_in_order_with_the_band_1_manifest_and_env_then_the_real_grade(self):
        out, err, calls, steps = self.run_loop(None)
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, (out[-1500:], err[-800:]))
        arm_calls = [c for c in calls if c['script'] == 'box']
        self.assertEqual([c['label'] for c in arm_calls], LABELS)
        for c in arm_calls:
            self.assertEqual(c['workload'], 'q18a-native41-progress.json', c['label'])
            self.assertEqual(c['env'], dict(READINESS='/r/ready', GEN='/o/S5-OVERLAP-GENERATION.json', S5_OVERLAY='/o',
                                            S5_LAUNCHER_NAME='l.sh', PACKAGE_MANIFEST='/o/SHA256SUMS',
                                            GENERATION='/o/S5-OVERLAP-GENERATION.json'))
        self.assertEqual([c['out'] for c in calls if c['script'] == 'checks' and c['action'] == 'full-idle'],
                         ['entry-%s.json' % x for x in LABELS] + ['end-full-idle.json'])
        for name in LABELS:
            self.assertEqual(steps['arm-' + name], 'PASS')
            watch = json.loads((self.q / ('engine-watch-arm-%s.json' % name)).read_text())
            self.assertEqual((watch['verdict'], watch['client_rc'], watch['container']), ('CLIENT_EXIT', 0, 'qwen-flash-prod'))
        self.assertEqual(steps['alpha-c0'], 'PASS')
        result = self.grade_file()
        self.assertEqual((result['verdict'], result['missing']), ('PASS', []))
        self.assertEqual({v['engine_watch'] for v in result['validity'].values()}, {'CLIENT_EXIT'})
        self.assertEqual(re.findall(r'^## (.+)$', (self.q / 'alpha-c0-report.md').read_text(), flags=re.M),
                         ['Mechanism and numerics', 'Performance', 'Release readiness'])

    def test_budget_before_an_arm_is_not_run_and_inconclusive(self):
        out, err, calls, steps = self.run_loop(box.ARM_ORDER[:5], jump_after='ingress-serial-live-1')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box'], LABELS[:5])
        for name in LABELS[5:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
            self.assertFalse((self.q / ('entry-%s.json' % name)).exists())
        self.assertIn('END FAILS=0 INCS=4 CRASHED=0', out, out[-800:])       # the three arms and the grade
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('declared-arm-missing:serial-solo-1', result['missing'])

    def test_a_dead_engine_stops_the_run_and_keeps_the_earlier_arms(self):
        started = time.monotonic()
        out, err, calls, steps = self.run_loop(box.ARM_ORDER[:5], die_in='local-live-1')
        self.assertLess(time.monotonic() - started, 120)
        self.assertIn('CRASHED=1 SOURCE=engine-death', out, out[-800:])
        self.assertEqual(steps['arm-local-live-1'], 'FAIL')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box'], LABELS[:6])
        for name in LABELS[6:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
        watch = json.loads((self.q / 'engine-watch-arm-local-live-1.json').read_text())
        self.assertEqual(watch['verdict'], 'DEAD')
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('local-live-1:engine-dead', result['failures'])
        self.assertEqual((result['validity']['local-live-1']['verdict'], result['validity']['local-live-1']['classification']),
                         ('FAIL', 'ENGINE'))
        self.assertEqual({result['validity'][x]['verdict'] for x in LABELS[:5]}, {'PASS'})
        self.assertEqual(sorted(result['summary']['arm_verdicts']), sorted(LABELS[:5]))


# --- commands, derivation, window ----------------------------------------------------------------------

ARGPARSE_VALUES = dict(tap.ARGPARSE_VALUES, QNAME='q18a-alpha-c0', RUN_ID='q18a-alpha-c0-fixture', _mode='local', _cohort='live',
                       _pair='1', _label='local-live-1')


def bash_words(args, draft, values=None):
    budget = tap.le_fix_function(draft, 'native41_budget')
    values = dict(ARGPARSE_VALUES, **(values or {}))
    import shlex
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(HERE))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    lines += ['GRADE_ARGS=(--arm "$Q/arm-fused-live-2.json" --ingress "fused-live-2=$Q/ingress-fused-live-2.json" '
              '--watch "fused-live-2=$Q/engine-watch-arm-fused-live-2.json")',
              'source "$H/runs/q18a_alpha_c0_settings.sh" || exit 97', budget, "printf '%s\\0' " + args]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode().split('\0')[:-1], done.stderr.decode()[-400:]


class Commands(unittest.TestCase):
    def test_every_body_command_parses_with_its_module(self):
        import e3_native39_trunk_checks as checks
        modules = {'e3_alpha_c0_box': box, 'e3_native39_trunk_checks': checks}
        rows = body_commands(queue.derive()[DRAFT])
        self.assertEqual(sorted({m for m, _ in rows}), sorted(modules))
        actions = []
        original = argparse.ArgumentParser.parse_args

        def capture(parser, args=None, namespace=None):
            raise held.Parsed(original(parser, args, namespace))
        for module, args in rows:
            argv = held.expand(args, dict(tap.QUEUE_VALUES, _mode='local', _cohort='live', _pair='1', _label='local-live-1',
                                          _workload='/h/workloads/q18a-native41-progress.json'))
            with mock.patch.object(argparse.ArgumentParser, 'parse_args', capture):
                with self.assertRaises(held.Parsed) as parsed, contextlib.redirect_stderr(io.StringIO()):
                    modules[module].main(argv)
            actions.append((module, argv[0], vars(parsed.exception.args[0])))
        self.assertEqual([a[1] for a in actions], ['instruments', 'full-idle', 'arm', 'ingress', 'instruments', 'full-idle', 'grade'])
        arm = next(ns for m, a, ns in actions if a == 'arm')
        self.assertEqual((arm['mode'], arm['cohort'], arm['index'], arm['control_timeout_s']), ('local', 'live', 1, 60.0))
        grade = actions[-1][2]
        self.assertEqual((grade['out'], grade['report']), ('/q/alpha-c0-grade.json', '/q/alpha-c0-report.md'))
        self.assertEqual(grade['log'], ['/q/engine-stream.log', '/q/engine-stream.err'])
        ingress = next(ns for m, a, ns in actions if a == 'ingress')
        self.assertEqual(ingress['generation'], progress.base.ARM_GENERATIONS)

    def test_the_loop_list_is_the_lead_order_and_the_split_runs_in_bash(self):
        draft = queue.derive()[DRAFT]
        (loop,) = re.findall(r'^\s*for _label in ([a-z0-9 -]+); do$', draft, flags=re.M)
        self.assertEqual(loop.split(), LABELS)
        self.assertEqual(list(queue.ARMS), LABELS)
        self.assertEqual(queue.ARMS, ('serial-solo-0', 'local-solo-0', 'local-live-0', 'serial-live-0',
                                      'serial-live-1', 'local-live-1', 'local-solo-1', 'serial-solo-1'))
        split = '    IFS=- read -r _mode _cohort _pair <<< "$_label"'
        self.assertIn(split, draft)
        script = 'for _label in %s; do\n%s\nprintf "%%s %%s %%s\\n" "$_mode" "$_cohort" "$_pair"\ndone' % (loop, split)
        done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
        self.assertEqual([tuple(x.split()) for x in done.stdout.splitlines()], [(m, c, str(i)) for m, c, i in box.ARM_ORDER])


class QueueArgparse(unittest.TestCase):
    def test_every_queue_python_call_parses_with_its_real_parser(self):
        texts = queue.derive()
        draft = texts[DRAFT]
        calls = held.queue_python_calls(texts, (HERE / 'qlib.sh').read_text())
        scripts = sorted({c[1] for c in calls})
        self.assertEqual(scripts, sorted([
            'SETCFG', 'e2_memory_attribution.py', 'e3_native28_host_observer.py', 'e3_native30_e_grade.py',
            'e3_native39_trunk_checks.py', 'e3_native39_w2_length.py', 'e3_alpha_c0_box.py',
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
        # Alpha-perf's calls one for one: only the body's client and grade changed module.
        parent = held.queue_python_calls(tap.queue.derive(), (HERE / 'qlib.sh').read_text())
        self.assertEqual(len(rows), len(parent))
        ARGPARSE_RECEIPTS[:] = rows

        def first(script, action):
            return next(a for n, s, a, f in calls if s == script and a.split()[0] == action)
        arm, grade = first('e3_alpha_c0_box.py', 'arm'), first('e3_alpha_c0_box.py', 'grade')
        for label, original, (old, new) in (('arm-unknown-option', arm, ('--control-timeout-s', '--control-timeout-sec')),
                                            ('arm-undeclared-mode', arm, ('--mode "$_mode"', '--mode burst')),
                                            ('arm-undeclared-round', arm, ('--index "$_pair"', '--index 6')),
                                            ('grade-without-out', grade, ('--out "$Q/alpha-c0-grade.json"', ''))):
            with self.subTest(control=label):
                self.assertEqual(original.count(old), 1)
                rc, words, err = bash_words(original.replace(old, new), draft)
                self.assertEqual(rc, 0, err)
                self.assertEqual(held.real_parse(HERE / 'e3_alpha_c0_box.py', words)[0], 'refused', label)


class Derivation(unittest.TestCase):
    def rename_old(self, text):
        return queue.rename(text)

    def test_derived_texts_are_on_disk_and_parse(self):
        derived = queue.derive()
        for name, text in derived.items():
            self.assertEqual((ROOT / name).read_text(), text, name)
            self.assertEqual(subprocess.run(['bash', '-n', str(ROOT / name)], capture_output=True).returncode, 0, name)
        settings = derived[SETTINGS]
        extra = [line for line in settings.splitlines() if line.strip().startswith('"EXTRA_ENV=')]
        self.assertEqual(len(extra), 1)
        words = extra[0].split()
        for flag in ('FLASH_S5_AUTO_ARM=1', 'FLASH_S5_NATIVE_PROGRESS=burst', 'FLASH_S5_NATIVE_QSA_POLICY=dense',
                     'FLASH_S5_STAGE_LOCAL=1', 'FLASH_S5_REPLAY_OBSERVER=0', 'FLASH_S5_ENGINE_OFFERS=0'):
            self.assertEqual(words.count(flag), 1, flag)
        for absent in ('FLASH_S5_UNION_RETIRE', 'FLASH_S5_QSA_FUSED', 'FLASH_S5_STAGE_OVERLAP', 'FLASH_S5_NATIVE_QSA_TRACE', 'FLASH_S5_NATIVE_QSA_STOP_AFTER'):
            self.assertNotIn(absent, extra[0])

    def test_settings_add_that_token_and_its_comment_nothing_else(self):
        old = self.rename_old((ROOT / 'harness/runs/q18a_alpha_perf_settings.sh').read_text())
        new = queue.derive()[SETTINGS]
        diff = [line for line in difflib.ndiff(old.splitlines(), new.splitlines()) if line[:1] in '+-']
        self.assertEqual([line[2:].replace(' ' + queue.SWITCH, '') for line in diff if line[0] == '+' and 'EXTRA_ENV' in line],
                         [line[2:] for line in diff if line[0] == '-'])
        self.assertEqual([line[2:] for line in diff if line[0] == '+' and 'EXTRA_ENV' not in line], [queue.SETTINGS_NOTE.strip()])

    def test_the_declaration_adds_one_assert_and_the_pins(self):
        old = self.rename_old((ROOT / 'harness/runs/q18a_alpha_perf_declare.sh').read_text())
        new = queue.derive()[DECLARE]
        diff = [line for line in difflib.ndiff(old.splitlines(), new.splitlines()) if line[:1] in '+-']
        self.assertEqual(len([x for x in diff if x[0] == '-']), 1)
        self.assertIn('+ ' + queue.DECLARE_SWITCH.strip(), diff)
        self.assertIn('e3_alpha_c0_box.py alpha_c0_gate.py', new)

    def test_everything_kept_is_alpha_perf_line_for_line(self):
        old = self.rename_old((ROOT / 'harness/runs/q18a_alpha_perf.draft.sh').read_text())
        body_old = self.rename_old(tap.queue.BODY)
        self.assertEqual(old.count(body_old), 1)
        new = queue.derive()[DRAFT]
        self.assertEqual(new.count(queue.BODY), 1)
        old_lines, new_lines = old.replace(body_old, '').splitlines(), new.replace(queue.BODY, '').splitlines()
        diff = [line for line in difflib.ndiff(old_lines, new_lines) if line[:1] in '+-']
        removed = [line[2:] for line in diff if line[0] == '-']
        added = [line[2:] for line in diff if line[0] == '+']
        # Every removed line names alpha-perf's gate or grade; each has its alpha-c0 line.
        self.assertEqual(len(removed), 7, removed)
        for line in removed:
            self.assertTrue('alpha_perf_gate' in line or 'alpha-perf-grade' in line, line)
            self.assertTrue(any('alpha_c0_gate' in x or 'alpha-c0-grade' in x for x in added), line)
        declared = queue.HEADER_NOTE.splitlines() + [queue.DRAFT_SWITCH.strip(), '  e3_alpha_c0_box.py alpha_c0_gate.py; do',
                                                     queue.INTERFACE_NEW.splitlines()[0]]
        self.assertEqual([x for x in added if not ('alpha_c0_gate' in x or 'alpha-c0-grade' in x) and x not in declared], [], diff)

    def test_the_draft_reads_alpha_c0_only(self):
        draft = queue.derive()[DRAFT]
        for text in ('from alpha_c0_gate import window_ok', 'from alpha_c0_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,PHASES',
                     "perf=p.parent/'alpha-c0-grade.json'", 'python3 "$H/e3_alpha_c0_box.py" arm', 'python3 "$H/e3_alpha_c0_box.py" grade',
                     "assert env.get('FLASH_S5_STAGE_LOCAL') == '1'"):
            self.assertIn(text, draft)
        for text in ('from alpha_perf_gate import', 'python3 "$H/e3_alpha_perf_box.py"', 'q18a_alpha_perf', 'alpha-perf-grade'):
            self.assertNotIn(text, draft)
        self.assertEqual(queue.BODY.count('engine_watched '), 3)
        self.assertEqual(len(re.findall(r'^\s*python3 "\$H/', queue.BODY, flags=re.M)), 0)
        self.assertIn(tap.queue.GUARD_NEW, draft.replace('alpha_c0', 'alpha_perf').replace('q18a-alpha-c0', 'q18a-alpha-perf')
                      if tap.queue.GUARD_NEW not in draft else draft)

    def test_the_arm_loop_keeps_the_progress_lines_flowing(self):
        # The lead's post-run tool reads the same followed engine logs the grade does: the follow is alpha-perf's text.
        draft = queue.derive()[DRAFT]
        old = (ROOT / 'harness/runs/q18a_alpha_perf.draft.sh').read_text()
        for line in old.splitlines():
            if 'engine-stream.log' in line and 'grade' not in line and 'alpha' not in line:
                self.assertIn(line, draft)
        self.assertEqual(progress.PROGRESS_LINE, 'flash-s5 progress ')
        env = [line for line in queue.derive()[SETTINGS].splitlines() if '"EXTRA_ENV=' in line][0]
        self.assertIn('FLASH_S5_NATIVE_PROGRESS=burst', env)


class Window(unittest.TestCase):
    def test_the_box_budget_and_the_arm_entry_rule(self):
        import native40_box_gate as n40
        import native41_box_gate as n41
        self.assertEqual((gate.CLEANUP_S, gate.NOT_BEFORE), (n40.CLEANUP_S, n40.NOT_BEFORE))
        # C0's own hard end, 2026-11-02T09:30Z = production time 07:00Z plus the 9,000 s budget: the operator's instruction of 2 Nov
        # (about 03:35 CET); the lead approved this pin edit (the latest-start string below follows from it). native40's stays 06:00Z.
        self.assertEqual((gate.HARD_END, n40.HARD_END), (1793611800.0, 1793599200.0))
        for name in n41.KEPT:
            self.assertEqual(gate.PHASES[name], n40.PHASES[name], name)
        self.assertEqual(gate.ACTIVE_S + gate.CLEANUP_S, gate.BOX_BUDGET_S)
        self.assertEqual(gate.BOX_BUDGET_S, 9000)                # lead's go, 2 Nov: 5,400 s let only arms 1-3 enter
        self.assertEqual((gate.ARMS, len(queue.ARMS), gate.BAND1_ARMS), (8, 8, 8))
        self.assertEqual(gate.ARM_RESERVE_S, 450 + 1500 + 30 + 60)
        self.assertEqual(gate.LAST_ARM_ENTRY_S + gate.ARM_RESERVE_S + gate.TAIL_RESERVE_S, gate.ACTIVE_S)
        self.assertTrue(gate.arm_fits(gate.LAST_ARM_ENTRY_S))
        self.assertFalse(gate.arm_fits(gate.LAST_ARM_ENTRY_S + 1))
        self.assertLess(gate.EXPECTED_RUNTIME_S['nominal'], gate.BOX_BUDGET_S)
        body = queue.BODY
        for text, count in (('timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments', 2),
                            ('timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle', 2),
                            ('timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_alpha_c0_box.py" arm', 1),
                            ('timeout 60s python3 "$H/e3_native39_trunk_checks.py" ingress', 1),
                            ('timeout 600s python3 "$H/e3_alpha_c0_box.py" grade', 1)):
            self.assertEqual(body.count(text), count, text)
        self.assertEqual((n41.INSTRUMENTS_CAP_S, n41.FULL_IDLE_CAP_S, n41.INGRESS_CAP_S, gate.GRADE_CAP_S), (120, 450, 60, 600))
        self.assertEqual(gate.latest_arm(), gate.HARD_END - gate.BOX_BUDGET_S)
        self.assertEqual(time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime(gate.latest_arm())), '2026-11-02T07:00Z')
        self.assertTrue(gate.window_ok(gate.latest_arm()))
        self.assertFalse(gate.window_ok(gate.latest_arm() + 1))
        self.assertFalse(gate.window_ok(gate.NOT_BEFORE - 1))
        func = tap.le_fix_function(queue.derive()[DRAFT], 'native41_budget')
        for key, value in (('active', gate.ACTIVE_S), ('cleanup', gate.CLEANUP_S), ('health', gate.PHASES['health']),
                           ('end_identity', gate.PHASES['end_identity']), ('last_arm_entry', gate.LAST_ARM_ENTRY_S)):
            done = subprocess.run(['bash', '-c', func + 'native41_budget ' + key],
                                  env=dict(os.environ, HARNESS_DIR=str(HERE)), capture_output=True, text=True)
            self.assertEqual((done.returncode, done.stdout.strip()), (0, str(value)), done.stderr)

    def test_draft_window_slot_executes_at_real_wall_clock(self):
        draft = queue.derive()[DRAFT]
        start = draft.index("python3 - <<'PY_SLOT'")
        slot = draft[start:draft.index('\nPY_SLOT\n', start) + len('\nPY_SLOT\n')]
        self.assertIn('alpha_c0_gate', slot)
        done = subprocess.run(['bash', '-c', 'fail_early() { return 2; }\n' + slot],
                              env=dict(os.environ, HARNESS_DIR=str(HERE)), capture_output=True, text=True)
        expected = 0 if gate.window_ok(time.time()) else 2
        self.assertEqual(done.returncode, expected, done.stderr)
        SLOT_RECEIPTS.append(dict(draft='q18a_alpha_c0.draft.sh', real_wall_clock=True,
                                  verdict='PASS' if done.returncode == 0 else 'OUTSIDE-WINDOW',
                                  active_s=gate.ACTIVE_S, cleanup_margin_s=gate.CLEANUP_S))


if __name__ == '__main__':
    unittest.main()
