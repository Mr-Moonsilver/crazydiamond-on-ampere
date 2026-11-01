"""Step B queue: the typed size over the real control plane, the grade over real emitters, the arm loop in bash. CPU only.

Classes:

* Arm / ArmNotInstalled: the derived draft's exact arm command line through two NativeRuntimeClients over loopback HTTP into the real
  endpoint, session, control client, driver, AutoArmPolicy, NativeProgress, FusedQsa and LayerGroups of a startup-bound stack. As in
  Step A's Arm, nothing links the HTTP prompt to the scheduler, so no native chunk runs here (a real group chunk on a connected Core
  is e3_step_b_probe.py in the engine self-run).
* Selection: the release's order and each defect of its receipts.
* Grade: sixteen arms of lines written by the real nm.ProgressTrace emitters (StepEmit), memory receipts in the engine's format:
  verdict per arm and overall, HARNESS and ENGINE, group engagement both ways, the R3 read, the report's sections.
* ArmLoop: the derived body under real bash with the draft's own functions, the real death watch and the real grade, over stub arm
  and check clients: the order, the budget, the size 4 trim, a dead engine.
* Commands / QueueArgparse: every python3 call of the derived texts parses with its own module's real parser.
* Derivation / Window: derived texts on disk and parse; the flag; everything else is Step A line for line; the budget.

Fixture leaves: step_b_fixture_leaves.json, beside the inherited files it names.
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
import shlex
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
import e3_step_a_box as sa  # noqa: E402
import e3_step_b_box as box  # noqa: E402
import step_a_gate as a_gate  # noqa: E402
import step_b_gate as gate  # noqa: E402
import step_b_queue as queue  # noqa: E402
import native_auto_arm as auto  # noqa: E402
import test_alpha_perf_queue as tap  # noqa: E402
import test_e3_native_overlap as native  # noqa: E402
import test_native38_plain_bind as native38  # noqa: E402
import test_native41_box_queue as held  # noqa: E402
import test_native41_progress_queue as hf  # noqa: E402
import test_step_a_queue as sq  # noqa: E402

progress = box.progress
ap = box.ap
SLOT_RECEIPTS = []
ARGPARSE_RECEIPTS = []
KEY = held.KEY
DRAFT = 'harness/runs/q18a_step_b.draft.sh'
SETTINGS = 'harness/runs/q18a_step_b_settings.sh'
DECLARE = 'harness/runs/q18a_step_b_declare.sh'
LABELS = box.required_labels()
ARM_ACTIONS = ['status', 'pause', 'native_progress', 'native_qsa_fused', 'native_layer_group', 'release_auto_arm', 'resume']
ALL = {'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst', 'FLASH_S5_QSA_FUSED': '1', 'FLASH_S5_LAYER_GROUPS': '1'}


def attach(engine_client, values):
    env = {'FLASH_S5_OVERLAP': '1', 'VLLM_API_KEY': KEY}
    env.update(values)
    app = held.App(engine_client)
    with held.fastapi_leaf():
        state = sq.pc.attach_native_router(SimpleNamespace(api_key=None), app, env=env)
    (path, endpoint, methods), = app.routers[0].routes
    assert (path, methods) == (sq.pc.NATIVE_PATH, ('POST',))
    return app, endpoint, state


def body_commands(text):
    body = text[text.index(queue.BODY.splitlines()[0]):]
    body = body[:body.index('record step-b ')]
    joined = re.sub(r'\\\n\s*', ' ', body)
    rows = []
    for line in joined.splitlines():
        match = re.search(r'python3 "\$H/([a-z0-9_]+)\.py"(.*)$', line)
        if match:
            rows.append((match.group(1), re.split(r'\s+>\s*"', match.group(2))[0].strip()))
    return rows


def arm_argv(**values):
    (args,) = [a for m, a in body_commands(queue.derive()[DRAFT]) if m == 'e3_step_b_box' and a.startswith('arm ')]
    values = dict(tap.QUEUE_VALUES, **values)
    values['_label'] = '%s-%s-%s' % (values['_mode'], values['_cohort'], values['_pair'])
    values['_mode#g'] = values['_mode'][1:]
    return held.expand(args, values)


# --- arms over the real control plane -----------------------------------------------------------------

class Arm(unittest.TestCase):
    INSTALL = True

    def setUp(self):
        self._logger = sys.modules.get('vllm.logger')
        sys.modules['vllm.logger'] = SimpleNamespace(init_logger=logging.getLogger)
        self.stack, self.core = native38.OwnerStartup().stack()
        held.pinned_reads(self.stack.scheduler)
        self.ready = native38.bootstrap_full_startup(self.core, self.stack.driver, timeout_s=10)
        self.driver = self.stack.driver
        # s5_startup.install_engine's order: the auto-arm policy, progress, the fused switch, the group switch.
        self.policy = auto.install(self.core, self.driver, {auto.ENV: '1'})
        self.progress = self.driver.install_progress('burst')
        self.fused = self.driver.install_qsa_fused()
        self.groups = self.driver.install_layer_groups() if self.INSTALL else None
        self.env = mock.patch.dict(os.environ, dict(ALL, VLLM_API_KEY=KEY))
        self.env.start()
        self.loop = held.EngineLoop(self.core)
        app, endpoint, _ = attach(native.Frontend(self.core), ALL)
        self.server = hf.ProgressServer(self.loop, app, endpoint).__enter__()
        self.tmp = tempfile.TemporaryDirectory()

    tearDown = hf.Arm.tearDown
    latch = hf.Arm.latch

    def run_arm(self, selection, cohort, index, name):
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

    def test_every_size_in_both_bands_through_the_exact_cli(self):
        salts = {}
        before = progress.label, progress.release, progress.progress_reasons, progress.progress_view, progress.ACTIONS
        actions_before = progress.base.ARM_ACTIONS
        for selection, cohort, index, name, budget, band, previous in (
                ('g2', 'live', 2, 'q18a-alpha-perf-band2', 32768, 2, 1),
                ('g1', 'live', 2, 'q18a-alpha-perf-band2', 32768, 2, 2),
                ('g4', 'solo', 0, 'q18a-native41-progress', 16384, 1, 1),
                ('g2', 'solo', 1, 'q18a-native41-progress', 16384, 1, 4)):
            size = box.size_of(selection)
            with self.subTest(selection=selection, cohort=cohort, index=index):
                self.latch()
                rc, record, actions, served = self.run_arm(selection, cohort, index, name)
                self.assertEqual(rc, 2)      # No native chunk here, by construction: incomplete by its own receipts.
                self.assertNotIn('error', record, record.get('error'))
                self.assertEqual((record['label'], record['selection'], record['size'], record['mode'], record['band'], record['workload']),
                                 (box.label(selection, cohort, index), selection, size, 'burst', band, name))
                self.assertEqual(actions[:7], ARM_ACTIONS)
                self.assertEqual(set(actions), set(ARM_ACTIONS))
                self.assertEqual([actions.count(a) for a in ARM_ACTIONS[1:]], [1] * 6)
                self.assertEqual((record['selection_switch']['fused'], record['selection_switch']['route']), (True, 'full'))
                self.assertEqual((record['group_switch']['size'], record['group_switch']['previous'], record['group_switch']['route']),
                                 (size, previous, 'full'))
                self.assertEqual((self.fused.selected, self.groups.size, record['release']['released']), (True, size, True))
                self.assertEqual({s['native_layer_groups']['size'] for s in record['progress'] if s['label'] != 'entry'}, {size})
                self.assertEqual(record['group_delta'], dict.fromkeys(box.GROUP_COUNTERS, 0))
                self.assertEqual(record['group_delta_append'], dict.fromkeys(box.GROUP_COUNTERS, 0))
                self.assertEqual([c['max_tokens'] for c in served], [budget] * 8 + [1, 1])
                self.assertIn('progress-chunks-not-recorded', record['reasons'])
                self.assertNotIn('control-sequence-not-pause-native-progress-release-resume', record['reasons'])
                self.assertNotIn('arm-sent-a-non-receipt-action', record['reasons'])
                self.assertNotIn('control-sequence-not-pause-progress-fused-group-release-resume', record['reasons'])
                self.assertFalse([x for x in record['reasons'] if x.startswith('auto-not-proven:unexpected')], record['reasons'])
                salts[record['label']] = record['salt_sha256']
        self.assertEqual(len(set(salts.values())), 4)
        # The seams are this process's one arm; they are back after each.
        self.assertEqual((progress.label, progress.release, progress.progress_reasons, progress.progress_view, progress.ACTIONS),
                         before)
        self.assertIs(progress.base.ARM_ACTIONS, actions_before)

    def test_the_wrong_manifest_for_the_round_is_refused_before_any_traffic(self):
        rc, record, actions, served = self.run_arm('g2', 'live', 2, 'q18a-native41-progress')
        self.assertEqual((rc, record['reasons'], record['label']), (2, ['arm-refused'], 'g2-live-2'))
        self.assertEqual(record['error']['error_message'], 'alpha-perf-workload-declaration-invalid')
        self.assertEqual((actions, served), ([], []))
        self.assertEqual(self.groups.size, 1)

    def test_sizes_and_rounds_outside_the_declared_set_do_not_parse(self):
        for selection, index in (('g2', '3'), ('g3', '0'), ('g0', '0'), ('g8', '0')):
            argv = arm_argv(_mode=selection, _cohort='live', _pair=index, Q=self.tmp.name)
            with self.subTest(selection=selection), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                box.main(argv)
            self.assertFalse(Path(self.tmp.name, 'arm-%s-live-%s.json' % (selection, index)).exists())


class ArmNotInstalled(Arm):
    INSTALL = False

    def test_every_size_in_both_bands_through_the_exact_cli(self):
        self.latch()
        rc, record, actions, served = self.run_arm('g2', 'solo', 0, 'q18a-native41-progress')
        self.assertEqual((rc, record['complete'], served), (2, False, []))
        self.assertIn('arm-error', record['reasons'])
        self.assertEqual(record['error']['error_message'], 'native-layer-groups-not-installed')
        self.assertEqual(actions, ['status'])

    def test_the_wrong_manifest_for_the_round_is_refused_before_any_traffic(self):
        pass

    def test_sizes_and_rounds_outside_the_declared_set_do_not_parse(self):
        pass


class Selection(unittest.TestCase):
    """The arm's own proofs over declared client replies (leaf arm-control-replies), each defect its row."""

    class Client:
        def __init__(self, fused_reply=None, size_reply=None, route='full', group_route='full', released=True, raises=None):
            self.fused_reply, self.size_reply, self.route, self.group_route = fused_reply, size_reply, route, group_route
            self.released, self.raises = released, raises
            self.calls, self.sent = [], {}

        def pause(self):
            self.calls.append('pause')
            return dict(route='full', route_generation=1)

        def resume(self):
            self.calls.append('resume')
            return dict(resumed=True)

        def call(self, action, **fields):
            self.calls.append(action)
            self.sent[action] = fields
            if action == self.raises:
                raise ValueError('refused-by-the-engine')
            if action == 'native_progress':
                return dict(mode='burst', previous='current', installed='burst', route='full', route_generation=1, engine_epoch=7)
            if action == 'native_qsa_fused':
                fused = fields['fused'] if self.fused_reply is None else self.fused_reply
                return dict(fused=fused, previous=False, route=self.route, route_generation=1, engine_epoch=7)
            if action == 'native_layer_group':
                size = fields['size'] if self.size_reply is None else self.size_reply
                return dict(size=size, previous=1, route=self.group_route, route_generation=1, engine_epoch=7)
            return dict(released=self.released, engine_epoch=7)

    def test_order_and_the_replies_are_checked(self):
        client, record = self.Client(), {}
        box.release_groups(2)(client, record, 'burst')
        self.assertEqual(client.calls, ['pause', 'native_progress', 'native_qsa_fused', 'native_layer_group', 'release_auto_arm', 'resume'])
        self.assertEqual((client.sent['native_qsa_fused'], client.sent['native_layer_group']), (dict(fused=True), dict(size=2)))
        for client, text in ((self.Client(fused_reply=False), 'native-qsa-fused-selection-unproved'),
                             (self.Client(route='split'), 'native-qsa-fused-selection-unproved'),
                             (self.Client(size_reply=4), 'native-layer-group-selection-unproved'),
                             (self.Client(size_reply=2.0), 'native-layer-group-selection-unproved'),
                             (self.Client(group_route='split'), 'native-layer-group-selection-unproved'),
                             (self.Client(released=False), 'auto-arm-release-unproved')):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, text):
                box.release_groups(2)(client, {}, 'burst')
            self.assertEqual(client.calls[-1], 'resume')

    def test_a_refused_selection_still_resumes_and_never_releases(self):
        for action, calls in (('native_qsa_fused', ['pause', 'native_progress', 'native_qsa_fused', 'resume']),
                              ('native_layer_group', ['pause', 'native_progress', 'native_qsa_fused', 'native_layer_group', 'resume'])):
            client = self.Client(raises=action)
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, 'refused-by-the-engine'):
                box.release_groups(2)(client, {}, 'burst')
            self.assertEqual(client.calls, calls)

    def receipts(self, *, size=2, history=None, before=None, after=None, fused_before=None, fused_after=None):
        fused = dict(selected=True, selections=1, chunks=0, grants=0, fused_commands=0, run_ahead_polls=0, ledger_checks=0)
        block = dict(size=2, run_ahead=3, selections=1, chunks=0, plan_commands=0, group_commands=0, group_layers=0, group_layers_max=0)
        record = dict(control_history=['status'] + ARM_ACTIONS[1:] if history is None else history, progress=[
            dict(label='before-fresh', native_qsa_fused=dict(fused, **(fused_before or {})), native_layer_groups=dict(block, **(before or {}))),
            dict(label='after-fresh', native_qsa_fused=dict(fused, chunks=3, **(fused_after or {})),
                 native_layer_groups=dict(block, chunks=3, group_layers_max=2, **(after or {})))])
        return box.selection_reasons(record, size), record

    def test_selection_receipts(self):
        reasons, record = self.receipts()
        self.assertEqual((reasons, record['group_delta']['chunks'], record['fused_delta']['chunks'], record['group_max_after']), ([], 3, 3, 2))
        self.assertIn('group-size-not-this-arm', self.receipts(before=dict(size=4))[0])
        self.assertIn('group-size-not-this-arm', self.receipts(after=dict(size=1))[0])
        self.assertIn('group-size-not-this-arm', self.receipts(size=4)[0])
        self.assertIn('fused-not-selected-in-this-arm', self.receipts(fused_before=dict(selected=False))[0])
        self.assertIn('fused-not-selected-in-this-arm', self.receipts(fused_after=dict(selected=False))[0])
        self.assertIn('selection-moved-during-the-fresh-prompt', self.receipts(after=dict(selections=2))[0])
        self.assertIn('selection-moved-during-the-fresh-prompt', self.receipts(fused_after=dict(selections=2))[0])
        order = ['status', 'pause', 'native_progress', 'native_layer_group', 'native_qsa_fused', 'release_auto_arm', 'resume']
        self.assertIn('control-sequence-not-pause-progress-fused-group-release-resume', self.receipts(history=order)[0])
        twice = ARM_ACTIONS[:5] + ['native_layer_group'] + ARM_ACTIONS[5:]
        self.assertIn('native_layer_group-sent-other-than-once', self.receipts(history=twice)[0])
        none = [x for x in ARM_ACTIONS if x != 'native_qsa_fused']
        self.assertIn('native_qsa_fused-sent-other-than-once', self.receipts(history=none)[0])

    def test_missing_views_are_named_and_stop_there(self):
        record = dict(control_history=['status'] + ARM_ACTIONS[1:], progress=[dict(label='before-fresh'), dict(label='after-fresh')])
        self.assertEqual(box.selection_reasons(record, 2), ['fused-status-missing', 'group-status-missing'])
        self.assertNotIn('group_delta', record)


# --- grade --------------------------------------------------------------------------------------------

GPU = {'g1': {1: (21000, 22000, 900), 2: (21500, 22500, 500)}, 'g2': {1: (21100, 22100, 800), 2: (21600, 22600, 400)},
       'g4': {1: (21200, 22200, 700)}}


class StepEmit(hf.Emit):
    """hf.Emit plus one size's layer command rows (the real row layout, real emitters): no profile query, as every fused arm."""

    def chunk(self, start, end, *, size, live):
        self.txn += 1
        kinds = [('open', None)] + [('segment', 'layer')] * box.LAYER_COMMANDS_PER_CHUNK[size]
        for i, (kind, segment) in enumerate(kinds + [('release', None)]):
            if i:
                self.tick(self.CADENCE)
            self.command(kind, segment, 'wait' if kind == 'segment' and live and i <= 2 else 'tick', int(kind == 'segment'))
            if i == 1 and live:
                self.stock('execute_model')
                self.stock('sample_tokens')
                self.engine.record('w', self.seq, 'sample_tokens', self.now, self.now + 300_000, 2, 0)
        for text in self.engine.lines(txn=self.txn, request_generation=1, chunk_start=start, chunk_end=end,
                                      decode_live=live, mode='burst', open_rows=0):
            self.log(self.tick(100), 'engine', text)

    def memory(self, selection, band, rank, txn, start, end):
        allocated, reserved, free = GPU[selection][band]
        for point, retries, shift in (('open', 5 + txn, 0), ('released', 6 + txn, 1)):
            self.log(self.now, rank, 'flash-s5 memory point=%s rank=%d txn_seq=%d chunk=%d-%d allocated=%d reserved=%d '
                                     'inactive_split=%d alloc_retries=%d ooms=0 free=%d total=25296044032' % (
                                         point, rank, txn, start, end, (allocated + shift) * 2 ** 20, reserved * 2 ** 20,
                                         2 ** 26, retries, (free - shift) * 2 ** 20))


def synthetic_run(order=None, extra=None, *, rate=None, ratio=None):
    """The declared arms in order, each arm's real emitter lines, memory receipts, fused and group counters."""
    emit, arms, extra = StepEmit(), [], extra or {}
    for selection, cohort, index in (order or box.ARM_ORDER):
        size = box.size_of(selection)
        name, band = box.label(selection, cohort, index), ap.band_of_index(index)
        cuts = tap.band_cuts(band)
        start = emit.tick(5_000_000)
        emit.tick(500_000)
        chunks = len(cuts) - 1
        lives = [8] * chunks if cohort == 'live' else [8] + [0] * (chunks - 1)
        for c, live in enumerate(lives):
            for rank in range(4):
                emit.memory(selection, band, rank, c + 1, cuts[c], cuts[c + 1])
            emit.chunk(cuts[c], cuts[c + 1], size=size, live=live)
        if name in extra:
            emit.log(emit.now, 1, extra[name])
        end = emit.tick(1_000_000)
        factor = (ratio or {}).get(name, 1.0)
        streams = [x * factor for x in (40.0, 41.0, 42.0, 42.0, 43.0, 43.0, 44.0, 45.0)]
        window = {'aggregate_tokens_per_s': 340.0 * factor, 'stream_rates_tokens_per_s': streams, 'gap_ms_p99': 60.0, 'gap_ms_max': 100.0}
        client = {'prefill_tokens_per_s': (rate or {}).get(selection, 900.0), 'all_open_share_of_span': 1.0 if cohort == 'live' else 0.1}
        if cohort == 'live':
            client['decode_window'] = window
        commands = box.LAYER_COMMANDS_PER_CHUNK[size] * chunks
        delta = dict(selections=0, chunks=chunks, grants=12 * chunks, fused_commands=12 * chunks, run_ahead_polls=0, ledger_checks=50 * chunks)
        groups = dict.fromkeys(box.GROUP_COUNTERS, 0) if size == 1 else dict(
            selections=0, chunks=chunks, plan_commands=commands, group_commands=commands, group_layers=box.LAYERS_PER_CHUNK * chunks)
        arms.append(dict(label=name, selection=selection, size=size, mode='burst', cohort=cohort, index=index, band=band, complete=True,
                         reasons=[], wall_start_s=emit.wall(start), wall_end_s=emit.wall(end),
                         progress_delta={'chunks': chunks, 'commands': commands + 2 * chunks}, fused_delta=delta, group_delta=groups,
                         group_max_after=None if size == 1 else size, selection_switch=dict(fused=True, route='full'),
                         group_switch=dict(size=size, route='full'),
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

    def test_sixteen_arms_pass_with_all_sections(self):
        emit, arms = synthetic_run()
        result = self.grade(emit, arms)
        self.assertEqual((result['schema'], result['verdict'], result['missing'], result['failures'], result['classification']),
                         ('step-b-grade/1', 'PASS', [], [], None), result['missing'])
        self.assertEqual(result['required_arms'], LABELS)
        self.assertEqual(len(LABELS), 16)
        self.assertEqual({v['verdict'] for v in result['validity'].values()}, {'PASS'})
        row = result['table']['g2-live-2']
        self.assertEqual((row['band'], row['selection'], row['size'], row['mode'], row['cohort'], row['prompt_tokens'], row['chunks']),
                         (2, 'g2', 2, 'burst', 'live', 131094, 5))
        self.assertEqual((row['ttft_s'], row['s_per_chunk'], row['prefill_tok_s']), (5.0, 1.0, 900.0))
        self.assertEqual((row['min_stream_rate'], row['gap_p99_ms'], row['gap_max_ms'], row['commands_per_chunk']), (40.0, 60.0, 100.0, 26.0))
        self.assertAlmostEqual(row['p10_stream_ratio'], 40.7 / 50)
        self.assertEqual(result['table']['g1-live-2']['commands_per_chunk'], 50.0)
        self.assertEqual(result['table']['g4-live-0']['commands_per_chunk'], 14.0)
        engaged = result['validity']['g2-live-2']['engagement']
        self.assertEqual((engaged['engaged'], engaged['owned_chunks'], engaged['profile_max'], engaged['layer_commands_min'],
                          engaged['layer_commands_max']), (True, 5, 0, 24, 24))
        engaged = result['validity']['g1-live-2']['engagement']
        self.assertEqual((engaged['engaged'], engaged['layer_commands_max'], engaged['groups']['group_commands']), (True, 48, 0))
        self.assertEqual(result['validity']['g4-solo-0']['engagement']['layer_commands_min'], 12)
        band1, band2 = result['summary']['band1'], result['summary']['band2']
        self.assertEqual(sorted(band1['groups']), ['g1-live', 'g1-solo', 'g2-live', 'g2-solo', 'g4-live', 'g4-solo'])
        self.assertEqual(sorted(band2['groups']), ['g1-live', 'g1-solo', 'g2-live', 'g2-solo'])
        self.assertEqual(band2['groups']['g2-live']['arms'], ['g2-live-2'])
        self.assertEqual(band2['over_size_one']['g2-live']['prefill_tok_s'], 1.0)
        self.assertEqual(band1['over_size_one']['g4-live']['commands_per_chunk'], 14 / 50)
        self.assertNotIn('fused_over_stepped', band1)
        pair = result['memory_over_size_one']['g2-live-2']
        self.assertEqual((pair['group_minus_size_one']['peak_reserved_mib'], pair['group_minus_size_one']['min_free_mib']), (100.0, -100.0))
        self.assertEqual(sorted(result['memory_over_size_one']), sorted(x for x in LABELS if not x.startswith('g1')))
        self.assertNotIn('memory_fused_vs_stepped', result)
        self.assertEqual((result['speed_is_gate'], result['memory_is_gate']), (False, False))
        self.assertIn('Not assessed by this queue', result['release_readiness'])

    def test_the_seams_are_this_call_only(self):
        before = sa.required_labels, sa.engagement, sa.SELECTIONS, sa.ARM_ORDER, sa.memory_compare, sa.speed_compare
        emit, arms = synthetic_run(box.ARM_ORDER[:1])
        self.grade(emit, arms)
        self.assertEqual((sa.required_labels, sa.engagement, sa.SELECTIONS, sa.ARM_ORDER, sa.memory_compare, sa.speed_compare), before)
        self.assertEqual(sa.required_labels()[0], 'stepped-solo-0')
        with mock.patch.object(sa, 'read_nothing', None, create=True), self.assertRaises(RuntimeError), box.grade_seams():
            raise RuntimeError('inside')
        self.assertEqual(sa.required_labels()[0], 'stepped-solo-0')

    def test_speed_memory_and_r3_never_change_the_verdict(self):
        emit, arms = synthetic_run(rate=dict(g2=1.0), ratio={'g2-live-0': 0.5, 'g2-live-1': 0.5, 'g2-live-2': 0.5})
        result = self.grade(emit, arms)
        self.assertEqual(result['verdict'], 'PASS')
        read = result['r3_read']
        self.assertEqual((read['below_margin'], read['is_gate'], read['margin'], read['rounds']), (['0', '1', '2'], False, 0.03, 3))
        self.assertAlmostEqual(read['g2_minus_g1']['0'], -0.425)

    def test_the_r3_read_is_on_the_margin(self):
        table = {box.label('g2', 'live', 0): dict(decode_ratio=0.90), box.label('g1', 'live', 0): dict(decode_ratio=0.93),
                 box.label('g2', 'live', 1): dict(decode_ratio=0.89), box.label('g1', 'live', 1): dict(decode_ratio=0.93),
                 box.label('g1', 'live', 2): dict(decode_ratio=0.93)}
        read = box.r3_read(table)
        self.assertEqual(read['below_margin'], ['1'])
        self.assertIsNone(read['g2_minus_g1']['2'])
        self.assertEqual(read['rounds'], 2)

    def test_a_group_arm_that_did_not_engage_is_inc(self):
        emit, arms = synthetic_run()
        name = 'g2-live-0'
        groups = next(a['group_delta'] for a in arms if a['label'] == name)
        chunks = groups['chunks']
        plan = groups['plan_commands']
        for field, value, reason in (
                ('chunks', chunks - 1, 'group-chunks-delta-differs-from-the-auto-owned-chunks'),
                ('chunks', 0, 'group-chunks-delta-differs-from-the-auto-owned-chunks'),
                ('group_commands', plan - 1, 'group-commands-differ-from-the-plan'),
                ('group_commands', plan + 1, 'group-commands-differ-from-the-plan'),
                ('group_layers', groups['group_layers'] - 1, 'group-layers-differ-from-the-model')):
            with self.subTest(field=field, value=value):
                result = self.grade(emit, self.edited(arms, name, group_delta=dict(groups, **{field: value})))
                self.assertEqual((result['verdict'], result['failures']), ('INC', []))
                self.assertIn(name + ':not-engaged', result['missing'])
                row = result['validity'][name]
                self.assertEqual((row['verdict'], row['classification'], row['reasons']), ('INC', None, [reason]))
        result = self.grade(emit, self.edited(arms, name, group_delta=dict(groups, plan_commands=plan + 1, group_commands=plan + 1)))
        self.assertEqual(result['validity'][name]['reasons'], ['plan-commands-differ-from-the-size'])
        result = self.grade(emit, self.edited(arms, name, group_max_after=1))
        self.assertEqual(result['validity'][name]['reasons'], ['largest-group-is-not-the-size'])
        for broken in (dict(group_delta=None), dict(fused_delta=None), dict(size=None), dict(selection='g4')):
            result = self.grade(emit, self.edited(arms, name, **broken))
            self.assertEqual(result['validity'][name]['reasons'], ['engagement-receipts-missing'], broken)
            self.assertEqual(result['verdict'], 'INC')

    def test_fused_rules_hold_in_every_arm(self):
        emit, arms = synthetic_run()
        for name in ('g1-live-0', 'g2-live-0', 'g4-live-0'):
            fused = next(a['fused_delta'] for a in arms if a['label'] == name)
            for field, value, reason in (('chunks', fused['chunks'] - 1, 'fused-chunks-delta-differs-from-the-auto-owned-chunks'),
                                         ('fused_commands', fused['fused_commands'] - 1, 'fused-commands-below-twelve-per-chunk')):
                with self.subTest(name=name, field=field):
                    result = self.grade(emit, self.edited(arms, name, fused_delta=dict(fused, **{field: value})))
                    self.assertEqual(result['validity'][name]['reasons'], [reason])

    def test_a_size_one_arm_shows_no_group_counter(self):
        emit, arms = synthetic_run()
        name = 'g1-solo-1'
        for field in box.GROUP_COUNTERS:
            with self.subTest(field=field):
                delta = dict.fromkeys(box.GROUP_COUNTERS, 0)
                delta[field] = 1
                result = self.grade(emit, self.edited(arms, name, group_delta=delta))
                self.assertEqual(result['validity'][name]['reasons'], ['size-one-arm-moved-a-group-counter'])
                self.assertEqual(result['verdict'], 'INC')

    def test_commands_of_another_size_under_this_arms_receipts_are_inc(self):
        # Lines of the other size under the receipts of this one: the progress join reads what really ran.
        order = [(('g2' if s == 'g1' else 'g1' if s == 'g2' else s), c, i) for s, c, i in box.ARM_ORDER]
        other_emit, other = synthetic_run(order)
        _, arms = synthetic_run()
        relabel = {box.label(*old): box.label(*new) for old, new in zip(order, box.ARM_ORDER)}
        mine = {a['label']: a for a in arms}
        swapped = [dict(a, label=relabel[a['label']], selection=mine[relabel[a['label']]]['selection'], size=mine[relabel[a['label']]]['size'],
                        group_delta=mine[relabel[a['label']]]['group_delta'], group_max_after=mine[relabel[a['label']]]['group_max_after'])
                   for a in other]
        result = self.grade(other_emit, swapped)
        self.assertEqual(result['verdict'], 'INC')
        for name in ('g1-solo-0', 'g2-solo-0'):
            self.assertEqual(result['validity'][name]['reasons'], ['layer-commands-per-chunk-differ-from-the-size'], name)
        self.assertEqual(result['validity']['g4-solo-0']['verdict'], 'PASS')

    def test_profile_queries_in_any_arm_are_inc(self):
        emit, arms = synthetic_run()
        kinds = {'chunks': 3, 'profile': [12, 12, 12], 'layer': [48, 48, 48]}
        with mock.patch.object(sa, 'chunk_kinds', lambda row: kinds):
            row = box.engagement(next(a for a in arms if a['label'] == 'g1-solo-0'), {})
        self.assertIn('arm-sent-profile-queries', row['reasons'])

    def test_each_engine_signature_fails_that_arm_as_engine(self):
        for line, name in (('(EngineCore_DP0 pid=101) ERROR native driver failed: tp-ledger-mismatch stage=0', 'tp-ledger-mismatch'),
                           ('(Worker_TP1 pid=202) ERROR raise DriverError(tp-collective-stall)', 'tp-collective-stall'),
                           ('(EngineCore_DP0 pid=101) ERROR fused-profile-differs-from-the-admission', 'fused-profile'),
                           ('(Worker_TP1 pid=202) ERROR QSA layers disagree on the packed selection width', 'tp-disagree')):
            with self.subTest(name=name):
                emit, arms = synthetic_run(extra={'g2-live-2': line})
                result = self.grade(emit, arms)
                row = result['validity']['g2-live-2']
                self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))
                self.assertEqual((row['verdict'], row['classification']), ('FAIL', 'ENGINE'))
                self.assertIn(name, row['reasons'])
                self.assertEqual({v['verdict'] for k, v in result['validity'].items() if k != 'g2-live-2'}, {'PASS'})

    def test_engine_watch_ingress_incomplete_and_a_trimmed_arm(self):
        emit, arms = synthetic_run()
        name = 'g1-live-0'
        watches = {a['label']: tap.ALIVE for a in arms}
        result = self.grade(emit, arms, watches=dict(watches, **{name: {'verdict': 'DEAD'}}))
        self.assertEqual((result['verdict'], result['classification'], result['validity'][name]['crash_free']), ('FAIL', ['ENGINE'], False))
        kept = [a for a in arms if a['label'] != name]
        result = self.grade(emit, kept, watches=dict(watches, **{name: {'verdict': 'DEAD'}}))
        self.assertEqual((result['validity'][name]['reasons'], result['validity'][name]['classification']), (['engine-dead'], 'ENGINE'))
        # A trimmed size 4 repeat is INC, never PASS, and everything else keeps its row.
        kept = [a for a in arms if a['label'] not in ('g4-live-1', 'g4-solo-1')]
        kept_watch = {a['label']: tap.ALIVE for a in kept}
        result = self.grade(emit, kept, watches=kept_watch)
        self.assertEqual((result['verdict'], result['validity']['g4-live-1']['verdict']), ('INC', 'INC'))
        self.assertIn('declared-arm-missing:g4-solo-1', result['missing'])
        self.assertEqual(result['validity']['g2-live-2']['verdict'], 'PASS')
        watches.pop(name)
        result = self.grade(emit, arms, watches=watches)
        self.assertEqual((result['verdict'], result['validity'][name]['reasons']), ('INC', ['engine-watch-record-missing']))
        result = self.grade(emit, self.edited(arms, name, complete=False, reasons=['fresh-not-complete']))
        self.assertEqual((result['verdict'], result['validity'][name]['verdict']), ('INC', 'INC'))
        result = self.grade(emit, arms, ingress=dict({a['label']: tap.PASS for a in arms}, **{name: {'verdict': 'FAIL'}}))
        self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['HARNESS']))

    def test_held_f_completeness_defects_hold(self):
        emit, arms = synthetic_run()
        span = next(a for a in arms if a['label'] == 'g2-solo-2')
        inside = [i for i, (s, t) in enumerate(emit.lines) if s == 'err' and progress.PROGRESS_LINE in t
                  and span['wall_start_s'] <= progress.docker_stamp(t) <= span['wall_end_s']]
        lines = [x for i, x in enumerate(emit.lines) if i != inside[1]]
        result = self.grade(emit, arms, lines=lines)
        # One whole flush missing is one lost record under the declared rule: counted, within the limit.
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        self.assertEqual(result['validity']['g2-solo-2']['tears']['lost_records'], 1)
        self.assertIn('g2-live-2:band-not-the-round-index-band', self.grade(emit, self.edited(arms, 'g2-live-2', band=1))['missing'])
        result = self.grade(emit, arms, lines=[])
        self.assertIn('no-progress-lines-in-the-engine-log', result['missing'])

    def test_the_report_keeps_three_separate_sections(self):
        emit, arms = synthetic_run()
        result = self.grade(emit, arms)
        text = box.report(result)
        self.assertEqual(re.findall(r'^## (.+)$', text, flags=re.M), ['Mechanism', 'Performance', 'Release readiness'])
        self.assertTrue(text.startswith('# q18a-step-b report'))
        self.assertEqual(text.split('## Release readiness\n\n')[1].strip(), box.RELEASE_READINESS)
        mechanism = text[text.index('## Mechanism'):text.index('## Performance')]
        for name in LABELS:
            self.assertIn('| %s | PASS |' % name, mechanism)
        self.assertIn('group cmds', mechanism)
        self.assertRegex(mechanism, r'\| g2-solo-0 \| PASS .*\| 24-24 \|')
        self.assertRegex(mechanism, r'\| g4-solo-0 \| PASS .*\| 12-12 \|')
        performance = text[text.index('## Performance'):text.index('## Release readiness')]
        for field in ('### Band 1', '### Band 2', 'R3 read', 'not a verdict input', 'Device memory', 'cmds/chunk', 'min stream ratio', 'p10 stream ratio',
                      'gap p99 ms', 'gap max ms', 'prefill tok/s'):
            self.assertIn(field, performance)
        self.assertNotIn('PASS', performance)
        kept = [a for a in arms if a['label'] != 'g2-solo-2']
        text = box.report(self.grade(emit, kept))
        self.assertIn('Overall: **INC**', text)
        self.assertIn('| g2-solo-2 | not available |', text)

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
        self.assertEqual(out.getvalue().strip(), 'STEP_B_GRADE PASS')
        self.assertEqual(json.loads((q / 'grade.json').read_text())['schema'], box.GRADE_SCHEMA)
        self.assertEqual(box.main(['grade'] + args + ['--out', str(q / 'grade.json')]), 2)
        (q / 'arm-g2-solo-2.json').write_text('{not json')
        args[args.index(str(q / 'r.md'))] = str(q / 'r2.md')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(box.main(['grade'] + args + ['--out', str(q / 'grade-2.json')]), 2)
        self.assertIn('declared-arm-missing:g2-solo-2', json.loads((q / 'grade-2.json').read_text())['missing'])


# --- the derived arm loop under real bash ------------------------------------------------------------

STUB_BOX = tap.STUB_BOX.replace('e3_alpha_perf_box', 'e3_step_b_box').replace(
    "label = '%s-%s-%s' % (a[a.index('--mode') + 1],", "label = 'g%s-%s-%s' % (a[a.index('--size') + 1],")


class Tears(unittest.TestCase):
    """The pre-registered rule through Step B's grade, which is Step A's, over the two shapes of q18a-alpha-perf's box log:
    lines 3832 with 3835 (bytes moved to a later line) and 34985 (a header landing inside the JSON)."""
    FOREIGN = '(Worker_PP1_TP0_EP0 pid=124) INFO 11-01 20:06:28 [qsa_operation.py:797] flash-s5 native-alloc rank=2\n'

    def setUp(self):
        self.helper = Grade(methodName='grade')
        self.helper.setUp()
        self.addCleanup(self.helper.tmp.cleanup)
        real = tap.band_cuts

        def cuts(band):
            ends = real(band)
            step = (ends[-1] - ends[0]) // 16
            return [ends[0] + k * step for k in range(16)] + [ends[-1]]
        with mock.patch.object(tap, 'band_cuts', cuts):
            self.emit, self.arms = synthetic_run()

    def grade(self, lines=None, arms=None):
        return self.helper.grade(self.emit, arms or self.arms, lines=self.emit.lines if lines is None else lines)

    def worker_lines(self, name, rank):
        span = next(a for a in self.arms if a['label'] == name)
        return [i for i, (_, t) in enumerate(self.emit.lines)
                if progress.PROGRESS_LINE in t and '"side":"worker"' in t and '"rank":%d,' % rank in t
                and span['wall_start_s'] <= progress.docker_stamp(t) <= span['wall_end_s']]

    @staticmethod
    def cut(text):
        at = text.index('"rows":')
        return at + min(200, (len(text) - at) // 2)

    def spliced(self, line):
        stream, text = line
        at = self.cut(text)
        return stream, text[:at] + '(Worker_PP0_TP1_EP1 pid=102) ' + text[at:]

    def moved(self, line):
        stream, text = line
        at = self.cut(text)
        return (stream, text[:at] + self.FOREIGN), (stream, text.split(' ', 1)[0] + ' ' + text[at:])

    def tear(self, indexes, shape='spliced'):
        lines = list(self.emit.lines)
        for i in sorted(indexes, reverse=True):
            if shape == 'spliced':
                lines[i] = self.spliced(lines[i])
            else:
                lines[i:i + 1] = self.moved(lines[i])
        return lines

    def test_a_clean_run_counts_nothing(self):
        result = self.grade()
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        tears = result['progress']['tears']
        self.assertEqual((tears['torn_progress_lines'], tears['lost_flushes'], tears['lost_flushes_outside_every_arm']),
                         (0, 0, []))
        self.assertEqual((tears['lost_record_limit'], tears['dropped_chunk_limit']), (1, 2))
        for name in result['required_arms']:
            self.assertEqual(result['validity'][name]['tears']['lost_records'], 0)

    def test_one_torn_line_of_either_shape_is_counted_and_its_chunks_drop_out(self):
        clean = self.grade()['progress']['arms']['g2-solo-0']
        for shape in ('spliced', 'moved'):
            result = self.grade(self.tear([self.worker_lines('g2-solo-0', 3)[2]], shape))
            self.assertEqual(result['verdict'], 'PASS', (shape, result['missing']))
            row = result['progress']['arms']['g2-solo-0']
            tears = row['tears']
            self.assertEqual((tears['torn_progress_lines'], tears['lost_records'], tears['within_limits']), (1, 1, True), shape)
            self.assertTrue(1 <= len(tears['dropped_chunk_starts']) <= 2, tears)
            self.assertEqual({c['chunk_start'] for c in row['chunks'] if c.get('dropped_out')},
                             set(tears['dropped_chunk_starts']))
            self.assertEqual(len(row['chunks']), len(clean['chunks']), 'a dropped chunk stays listed')
            self.assertLess(row['counted_chunks'], clean['counted_chunks'])
            self.assertEqual(result['validity']['g2-solo-0']['verdict'], 'PASS')
            self.assertEqual(result['progress']['arms']['g1-solo-0']['tears']['lost_records'], 0)
            self.assertIn('### Torn lines and lost records', box.report(result))

    def test_two_lost_records_in_one_arm_go_inc_and_one_in_each_of_two_arms_does_not(self):
        a, b = self.worker_lines('g2-live-0', 1)[1], self.worker_lines('g2-live-0', 3)[5]
        result = self.grade(self.tear([a, b]))
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('progress-g2-live-0:lost-records-above-the-declared-limit:2>1', result['missing'])
        self.assertEqual(result['validity']['g2-live-0']['verdict'], 'INC')
        self.assertEqual(result['validity']['g2-solo-0']['verdict'], 'PASS')
        result = self.grade(self.tear([a, self.worker_lines('g1-live-0', 3)[5]]))
        self.assertEqual(result['verdict'], 'PASS', result['missing'])

    def test_a_contract_error_spliced_into_a_torn_line_fails_engine(self):
        for mark, name in (('tp-ledger-mismatch', 'tp-ledger-mismatch'), ('disagree', 'tp-disagree-in-a-torn-line')):
            i = self.worker_lines('g2-live-1', 2)[3]
            lines = list(self.emit.lines)
            stream, text = lines[i]
            at = self.cut(text)
            lines[i] = (stream, text[:at] + '(Worker_PP0_TP1_EP1 pid=102) ' + mark + ' ' + text[at:])
            result = self.grade(lines)
            self.assertEqual(result['verdict'], 'FAIL', mark)
            self.assertEqual(result['validity']['g2-live-1']['classification'], 'ENGINE', mark)
            self.assertIn(name, result['validity']['g2-live-1']['signatures'], mark)

    def test_a_stray_header_with_an_intact_body_is_not_a_tear(self):
        i = self.worker_lines('g2-solo-1', 0)[1]
        lines = list(self.emit.lines)
        stream, text = lines[i]
        at = text.index(progress.PROGRESS_LINE)
        lines[i] = (stream, text[:at] + '(Worker_PP0_TP1_EP1 pid=102) ' + text[at:])
        result = self.grade(lines)
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        self.assertEqual(result['progress']['tears']['torn_progress_lines'], 0)
        self.assertEqual(result['progress']['arms']['g2-solo-1']['tears']['lost_records'], 0)


class MemoryRead(unittest.TestCase):
    """The lead's memory read and the slow-rank report through the real grade, over the synthetic run's real-format receipts."""

    def setUp(self):
        self.helper = Grade(methodName='grade')
        self.helper.setUp()
        self.addCleanup(self.helper.tmp.cleanup)
        self.emit, self.arms = synthetic_run()

    def grade(self, lines=None):
        return self.helper.grade(self.emit, self.arms, lines=self.emit.lines if lines is None else lines)

    def memory_lines(self, name, point='released'):
        span = next(a for a in self.arms if a['label'] == name)
        return [i for i, (_, t) in enumerate(self.emit.lines)
                if 'flash-s5 memory point=%s ' % point in t and span['wall_start_s'] <= progress.docker_stamp(t) <= span['wall_end_s']]

    @staticmethod
    def lowered(text, mib):
        head, free = text.split(' free=', 1)
        value, rest = free.split(' ', 1)
        return head + ' free=%d ' % (int(value) - mib * 2 ** 20) + rest

    @staticmethod
    def torn(text, mark=''):
        at = text.index(' reserved=') + len(' reserved=') + 3
        return text[:at] + '(Worker_PP0_TP1_EP1 pid=102) ' + mark + text[at:]

    def test_a_clean_run_reads_and_reports(self):
        result = self.grade()
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        reads = result['memory_read']
        self.assertEqual(sorted(reads), sorted(LABELS))
        self.assertEqual({reads[x]['verdict'] for x in LABELS if x.startswith('g1')}, {'MET'})
        # The synthetic group arms hold 100 MiB less free than size 1: outside the 2 MiB margin, read NOT_MET, verdict unchanged.
        self.assertEqual(reads['g2-live-0']['verdict'], 'NOT_MET')
        self.assertEqual(reads['g2-live-0']['min_free_below_size_one_mib'], 100.0)
        self.assertEqual(result['validity']['g2-live-0']['verdict'], 'PASS')
        self.assertEqual((result['memory_read_is_gate'], result['memory_is_gate'], result['memory_free_margin_mib']), (False, False, 2.0))
        text = box.report(result)
        self.assertIn('### Memory read, per arm', text)
        self.assertIn('### Slow rank per stage', text)

    def test_a_torn_memory_line_holding_the_minimum_makes_the_read_inc(self):
        lines = [list(x) for x in self.emit.lines]
        i = self.memory_lines('g1-live-0')[2]
        lines[i][1] = self.lowered(lines[i][1], 500)
        result = self.grade(lines)
        self.assertEqual(result['memory_read']['g1-live-0']['min_free_mib'], 399.0, 'the lowered whole line is the minimum')
        lines[i][1] = self.torn(lines[i][1])
        result = self.grade(lines)
        read = result['memory_read']['g1-live-0']
        self.assertEqual((read['verdict'], read['established']), ('INC', False))
        self.assertIn('torn-memory-lines:1', read['reasons'])
        self.assertEqual(read['min_free_mib'], 899.0, 'the whole lines no longer show the minimum, so no MET may be read')
        self.assertEqual(result['memory_read']['g2-live-0']['verdict'], 'INC', 'its matched size 1 arm is not established')
        self.assertIn('size-one-arm-torn-memory-lines:1', result['memory_read']['g2-live-0']['reasons'])
        # The progress tolerance stays as it is: one lost record, within the limit, the arm still passes.
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        self.assertEqual(result['validity']['g1-live-0']['tears']['torn_memory_lines'], 1)
        self.assertEqual(result['memory_read']['g1-live-1']['verdict'], 'MET')

    def test_a_lost_memory_receipt_makes_the_read_inc(self):
        drop = self.memory_lines('g1-solo-1', 'open')[3]
        result = self.grade([x for i, x in enumerate(self.emit.lines) if i != drop])
        read = result['memory_read']['g1-solo-1']
        self.assertEqual(read['verdict'], 'INC')
        self.assertIn('memory-receipts-missing:1', read['reasons'])

    def test_a_torn_memory_line_is_scanned_for_ooms_and_contract_errors(self):
        lines = [list(x) for x in self.emit.lines]
        i = self.memory_lines('g1-solo-0')[1]
        lines[i][1] = self.torn(lines[i][1].replace(' ooms=0 ', ' ooms=1 '))
        result = self.grade(lines)
        read = result['memory_read']['g1-solo-0']
        self.assertEqual(read['verdict'], 'NOT_MET')
        self.assertIn('oom-count-in-a-torn-memory-line', read['oom_evidence'])
        self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))
        self.assertEqual((result['validity']['g1-solo-0']['verdict'], result['validity']['g1-solo-0']['classification']), ('FAIL', 'ENGINE'))
        self.assertIn('g1-solo-0:oom', result['failures'])
        lines = [list(x) for x in self.emit.lines]
        lines[i][1] = self.torn(lines[i][1], 'CUDA out of memory ')
        self.assertIn('oom-text-in-a-torn-memory-line', self.grade(lines)['memory_read']['g1-solo-0']['oom_evidence'])
        lines = [list(x) for x in self.emit.lines]
        lines[i][1] = self.torn(lines[i][1], 'tp-ledger-mismatch ')
        result = self.grade(lines)
        self.assertEqual((result['verdict'], result['validity']['g1-solo-0']['classification']), ('FAIL', 'ENGINE'))
        self.assertIn('tp-ledger-mismatch', result['validity']['g1-solo-0']['signatures'])

    def test_an_oom_in_an_intact_receipt_fails_the_arm_as_engine(self):
        lines = [list(x) for x in self.emit.lines]
        later = self.memory_lines('g2-live-1')[4:]
        for i in later:
            lines[i][1] = lines[i][1].replace(' ooms=0 ', ' ooms=1 ')
        result = self.grade(lines)
        self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))
        row = result['validity']['g2-live-1']
        self.assertEqual((row['verdict'], row['classification'], row['reasons'][0]), ('FAIL', 'ENGINE', 'oom:ooms:4'), 'the counter rose on each of the 4 ranks')
        self.assertIn('oom', row['signatures'])
        self.assertEqual(result['memory_read']['g2-live-1']['verdict'], 'NOT_MET')
        self.assertEqual(result['validity']['g1-live-1']['verdict'], 'PASS')
        # A min free below the margin alone is a read, not a failure: the clean run's group arms stay PASS.
        clean = self.grade()
        self.assertEqual((clean['verdict'], clean['memory_read']['g2-live-0']['oom_evidence'],
                          clean['memory_read']['g2-live-0']['min_free_breach']), ('PASS', [], True))

    def test_the_slow_rank_is_read_per_chunk_and_never_graded(self):
        result = self.grade()
        slow = result['slow_rank']
        self.assertFalse(slow['is_gate'])
        self.assertTrue(slow['chunk_identity'].startswith('exact'))
        row = slow['arms']['g2-live-0']['all']['0']
        counted = result['progress']['arms']['g2-live-0']['counted_chunks']
        self.assertEqual(row['chunks'], counted)
        # The emitter's rank 1 replies later than rank 0 on every command: it is the slower rank on every chunk.
        self.assertEqual(row['slower_rank_chunks'], {'1': counted})
        self.assertGreater(row['own_s']['slow_rank_median'], row['own_s']['pooled_median'])
        self.assertEqual(sorted(slow['arms']['g2-live-0']['bands']), sorted({c['band'] for c in result['progress']['arms']['g2-live-0']['chunks']}))
        # The synthetic commands are all stage 0; stage 1 has none to time and is left out, not zero.
        self.assertNotIn('1', slow['arms']['g2-live-0']['all'])
        changed = self.helper.grade(self.emit, self.arms)
        self.assertEqual(changed['verdict'], result['verdict'])


class ArmLoop(unittest.TestCase):
    """queue.BODY runs as written; arm and check clients are stubs (leaf arm-loop-clients), the watch and grade real."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='step-b-loop-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q, self.h, self.bin, self.arms = (self.root / x for x in ('q', 'h', 'bin', 'arms'))
        for d in (self.q, self.h, self.bin, self.arms):
            d.mkdir()
        for name in ('native43_engine_death.py', 'step_a_gate.py', 'step_b_gate.py', 'native39_trunk_gate.py', 'native40_box_gate.py',
                     'native41_box_gate.py'):
            (self.h / name).symlink_to(HERE / name)
        (self.h / 'e3_native39_trunk_checks.py').write_text(tap.STUB_CHECKS)
        (self.h / 'e3_step_b_box.py').write_text(STUB_BOX)
        self.tool('nvidia-smi', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')
        self.tool('docker', 'case "$*" in *State.Running*) echo true;; *) echo 2026-11-01T10:48:09.000000000Z;; esac')
        self.tool('systemctl', 'echo 0')
        (self.q / 'unarmed-stock-prefill.json').write_text(json.dumps(tap.PASS))

    tool = tap.ArmLoop.tool

    def run_loop(self, order, *, jump_after=None, jump_to='LAST_ARM_ENTRY_S + 10', die_in=None, timeout=300):
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
                       'record() { real_record "$@"; local rc=$?; [[ $1 == %s ]] && SECONDS=$((%s)); return $rc; }' % (jump_after, jump_to)]
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
        return json.loads((self.q / 'step-b-grade.json').read_text())

    def test_sixteen_arms_in_order_with_the_band_manifest_and_env_then_the_real_grade(self):
        out, err, calls, steps = self.run_loop(None)
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, (out[-1500:], err[-800:]))
        arm_calls = [c for c in calls if c['script'] == 'box']
        self.assertEqual([c['label'] for c in arm_calls], LABELS)
        for c in arm_calls:
            expected = 'q18a-native41-progress.json' if c['label'][-1] in '01' else 'q18a-alpha-perf-band2.json'
            self.assertEqual(c['workload'], expected, c['label'])
        self.assertEqual([c['out'] for c in calls if c['script'] == 'checks' and c['action'] == 'full-idle'],
                         ['entry-%s.json' % x for x in LABELS] + ['end-full-idle.json'])
        for name in LABELS:
            self.assertEqual(steps['arm-' + name], 'PASS')
        self.assertEqual(steps['step-b'], 'PASS')
        result = self.grade_file()
        self.assertEqual((result['verdict'], result['missing']), ('PASS', []))
        self.assertEqual(re.findall(r'^## (.+)$', (self.q / 'step-b-report.md').read_text(), flags=re.M),
                         ['Mechanism', 'Performance', 'Release readiness'])

    def test_past_the_trim_the_size_4_repeats_give_way_and_band_2_runs(self):
        order = [x for x in box.ARM_ORDER if box.label(*x) not in ('g4-live-1', 'g4-solo-1')]
        out, err, calls, steps = self.run_loop(order, jump_after='ingress-g2-live-1', jump_to='TRIM_G4_REPEAT_S + 10')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box'], [box.label(*x) for x in order])
        for name in ('g4-live-1', 'g4-solo-1'):
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
            self.assertFalse((self.q / ('entry-%s.json' % name)).exists())
        for name in ('g2-solo-1', 'g1-solo-1', 'g2-live-2', 'g1-live-2', 'g1-solo-2', 'g2-solo-2'):
            self.assertEqual(steps['arm-' + name], 'PASS', name)
        self.assertIn('NOT_RUN: size 4 repeat trimmed for band 2', out)
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('declared-arm-missing:g4-live-1', result['missing'])

    def test_the_trim_only_touches_the_size_4_repeats(self):
        # The same clock before round 0's size 4 arms leaves them in; they are not repeats.
        out, err, calls, steps = self.run_loop(None, jump_after='instruments-start', jump_to=str(gate.TRIM_G4_REPEAT_S + 10))
        trimmed = ('g4-live-1', 'g4-solo-1')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box'], [x for x in LABELS if x not in trimmed])
        self.assertEqual((steps['arm-g4-solo-0'], steps['arm-g4-live-0']), ('PASS', 'PASS'))
        self.assertEqual((steps['arm-g4-live-1'], steps['arm-g4-solo-1']), ('INCONCLUSIVE', 'INCONCLUSIVE'))

    def test_budget_before_an_arm_is_not_run_and_inconclusive(self):
        out, err, calls, steps = self.run_loop(box.ARM_ORDER[:13], jump_after='ingress-g2-live-2')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box'], LABELS[:13])
        for name in LABELS[13:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
            self.assertFalse((self.q / ('entry-%s.json' % name)).exists())
        self.assertIn('END FAILS=0 INCS=4 CRASHED=0', out, out[-800:])       # the three arms and the grade
        self.assertEqual(self.grade_file()['verdict'], 'INC')

    def test_a_dead_engine_stops_the_arm_and_keeps_what_ran(self):
        started = time.monotonic()
        out, err, calls, steps = self.run_loop(box.ARM_ORDER[:12], die_in='g2-live-2')
        self.assertLess(time.monotonic() - started, 150)
        self.assertIn('CRASHED=1 SOURCE=engine-death', out, out[-800:])
        self.assertEqual(steps['arm-g2-live-2'], 'FAIL')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box'], LABELS[:13])
        for name in LABELS[13:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('g2-live-2:engine-dead', result['failures'])
        self.assertEqual({result['validity'][x]['verdict'] for x in LABELS[:12]}, {'PASS'})


# --- commands, derivation, window ----------------------------------------------------------------------

ARGPARSE_VALUES = dict(tap.ARGPARSE_VALUES, QNAME='q18a-step-b', RUN_ID='q18a-step-b-fixture', _mode='g2', _cohort='live',
                       _pair='2', _label='g2-live-2')


def bash_words(args, draft, values=None):
    budget = tap.le_fix_function(draft, 'native41_budget')
    values = dict(ARGPARSE_VALUES, **(values or {}))
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(HERE))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    lines += ['GRADE_ARGS=(--arm "$Q/arm-g2-live-2.json" --ingress "g2-live-2=$Q/ingress-g2-live-2.json" '
              '--watch "g2-live-2=$Q/engine-watch-arm-g2-live-2.json")',
              'source "$H/runs/q18a_step_b_settings.sh" || exit 97', budget, "printf '%s\\0' " + args]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode().split('\0')[:-1], done.stderr.decode()[-400:]


class Commands(unittest.TestCase):
    def test_every_body_command_parses_with_its_module(self):
        import e3_native39_trunk_checks as checks
        modules = {'e3_step_b_box': box, 'e3_native39_trunk_checks': checks}
        rows = body_commands(queue.derive()[DRAFT])
        self.assertEqual(sorted({m for m, _ in rows}), sorted(modules))
        actions = []
        original = argparse.ArgumentParser.parse_args

        def capture(parser, args=None, namespace=None):
            raise held.Parsed(original(parser, args, namespace))
        for module, args in rows:
            argv = held.expand(args, dict(tap.QUEUE_VALUES, **{'_mode#g': '2'}, _mode='g2', _cohort='live', _pair='2', _label='g2-live-2',
                                          _workload='/h/workloads/q18a-alpha-perf-band2.json'))
            with mock.patch.object(argparse.ArgumentParser, 'parse_args', capture):
                with self.assertRaises(held.Parsed) as parsed, contextlib.redirect_stderr(io.StringIO()):
                    modules[module].main(argv)
            actions.append((module, argv[0], vars(parsed.exception.args[0])))
        self.assertEqual([a[1] for a in actions], ['instruments', 'full-idle', 'arm', 'ingress', 'instruments', 'full-idle', 'grade'])
        arm = next(ns for m, a, ns in actions if a == 'arm')
        self.assertEqual((arm['size'], arm['cohort'], arm['index'], arm['control_timeout_s']), (2, 'live', 2, 60.0))
        grade = actions[-1][2]
        self.assertEqual((grade['out'], grade['report']), ('/q/step-b-grade.json', '/q/step-b-report.md'))

    def test_the_loop_list_is_the_lead_order_and_the_split_runs_in_bash(self):
        draft = queue.derive()[DRAFT]
        (loop,) = re.findall(r'^\s*for _label in ([a-z0-9 -]+); do$', draft, flags=re.M)
        self.assertEqual(loop.split(), LABELS)
        self.assertEqual(list(queue.ARMS), LABELS)
        split = '    IFS=- read -r _mode _cohort _pair <<< "$_label"'
        self.assertIn(split, draft)
        script = 'for _label in %s; do\n%s\nprintf "%%s %%s %%s %%s\\n" "$_mode" "${_mode#g}" "$_cohort" "$_pair"\ndone' % (loop, split)
        done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
        self.assertEqual([tuple(x.split()) for x in done.stdout.splitlines()],
                         [(m, m[1:], c, str(i)) for m, c, i in box.ARM_ORDER])

    def test_every_size_and_cohort_is_matched(self):
        rounds = {}
        for selection, cohort, index in box.ARM_ORDER:
            rounds.setdefault((cohort, index), []).append(selection)
        for (cohort, index), selections in rounds.items():
            self.assertEqual(sorted(selections), ['g1', 'g2', 'g4'] if index < 2 else ['g1', 'g2'], (cohort, index))
        # ABBA: each size's two band 1 rounds mirror around the middle of the band.
        self.assertEqual(list(box.BAND1[:6]), [(s, c, 1 - i) for s, c, i in reversed(box.BAND1[6:])])


class QueueArgparse(unittest.TestCase):
    def test_every_queue_python_call_parses_with_its_real_parser(self):
        texts = queue.derive()
        draft = texts[DRAFT]
        calls = held.queue_python_calls(texts, (HERE / 'qlib.sh').read_text())
        scripts = sorted({c[1] for c in calls})
        self.assertEqual(scripts, sorted([
            'SETCFG', 'e2_memory_attribution.py', 'e3_native28_host_observer.py', 'e3_native30_e_grade.py',
            'e3_native39_trunk_checks.py', 'e3_native39_w2_length.py', 'e3_step_b_box.py',
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
        # Step A's calls one for one: only the body's client and grade changed module.
        parent = held.queue_python_calls(sq.queue.derive(), (HERE / 'qlib.sh').read_text())
        self.assertEqual(len(rows), len(parent))
        ARGPARSE_RECEIPTS[:] = rows

        def first(script, action):
            return next(a for n, s, a, f in calls if s == script and a.split()[0] == action)
        arm, grade = first('e3_step_b_box.py', 'arm'), first('e3_step_b_box.py', 'grade')
        for label, original, (old, new) in (('arm-unknown-option', arm, ('--control-timeout-s', '--control-timeout-sec')),
                                            ('arm-undeclared-size', arm, ('--size "${_mode#g}"', '--size 3')),
                                            ('arm-undeclared-round', arm, ('--index "$_pair"', '--index 6')),
                                            ('grade-without-out', grade, ('--out "$Q/step-b-grade.json"', ''))):
            with self.subTest(control=label):
                self.assertEqual(original.count(old), 1)
                rc, words, err = bash_words(original.replace(old, new), draft)
                self.assertEqual(rc, 0, err)
                self.assertEqual(held.real_parse(HERE / 'e3_step_b_box.py', words)[0], 'refused', label)


class Derivation(unittest.TestCase):
    def test_derived_texts_are_on_disk_and_parse(self):
        derived = queue.derive()
        for name, text in derived.items():
            self.assertEqual((ROOT / name).read_text(), text, name)
            self.assertEqual(subprocess.run(['bash', '-n', str(ROOT / name)], capture_output=True).returncode, 0, name)
        extra = [line for line in derived[SETTINGS].splitlines() if line.strip().startswith('"EXTRA_ENV=')]
        self.assertEqual(len(extra), 1)
        words = extra[0].split()
        for flag in ('FLASH_S5_AUTO_ARM=1', 'FLASH_S5_NATIVE_PROGRESS=burst', 'FLASH_S5_NATIVE_QSA_POLICY=dense',
                     'FLASH_S5_QSA_FUSED=1', 'FLASH_S5_LAYER_GROUPS=1', 'FLASH_S5_REPLAY_OBSERVER=0', 'FLASH_S5_ENGINE_OFFERS=0'):
            self.assertEqual(words.count(flag), 1, flag)
        for absent in ('FLASH_S5_UNION_RETIRE', 'FLASH_S5_NATIVE_QSA_TRACE', 'FLASH_S5_NATIVE_QSA_STOP_AFTER'):
            self.assertNotIn(absent, extra[0])

    def test_settings_add_that_token_and_its_comment_nothing_else(self):
        old = queue.rename((ROOT / 'harness/runs/q18a_step_a_settings.sh').read_text())
        new = queue.derive()[SETTINGS]
        diff = [line for line in difflib.ndiff(old.splitlines(), new.splitlines()) if line[:1] in '+-']
        self.assertEqual([line[2:].replace(' ' + queue.GROUPS, '') for line in diff if line[0] == '+' and 'EXTRA_ENV' in line],
                         [line[2:] for line in diff if line[0] == '-'])
        self.assertEqual([line[2:] for line in diff if line[0] == '+' and 'EXTRA_ENV' not in line], [queue.SETTINGS_NOTE.strip()])

    def test_the_declaration_adds_one_assert_and_the_pins(self):
        old = queue.rename((ROOT / 'harness/runs/q18a_step_a_declare.sh').read_text())
        new = queue.derive()[DECLARE]
        diff = [line for line in difflib.ndiff(old.splitlines(), new.splitlines()) if line[:1] in '+-']
        self.assertEqual(len([x for x in diff if x[0] == '-']), 1)
        self.assertIn('+ ' + queue.DECLARE_GROUPS.strip(), diff)
        self.assertIn('e3_step_b_box.py step_b_gate.py', new)

    def test_everything_kept_is_step_a_line_for_line(self):
        old = queue.rename((ROOT / 'harness/runs/q18a_step_a.draft.sh').read_text())
        body_old = queue.rename(sq.queue.BODY)
        self.assertEqual(old.count(body_old), 1)
        new = queue.derive()[DRAFT]
        self.assertEqual(new.count(queue.BODY), 1)
        old_lines, new_lines = old.replace(body_old, '').splitlines(), new.replace(queue.BODY, '').splitlines()
        diff = [line for line in difflib.ndiff(old_lines, new_lines) if line[:1] in '+-']
        removed = [line[2:] for line in diff if line[0] == '-']
        added = [line[2:] for line in diff if line[0] == '+']
        self.assertEqual(len(removed), 8, removed)
        for line in removed:
            self.assertTrue('step_a_gate' in line or 'step-a-grade' in line or 'last_arm_entry' in line, line)
        declared = queue.HEADER_NOTE.splitlines() + [queue.DRAFT_GROUPS.strip(), queue.INTERFACE_NEW]
        self.assertEqual([x for x in added if not ('step_b_gate' in x or 'step-b-grade' in x or 'trim_g4_repeat' in x) and x not in declared], [], diff)

    def test_the_draft_reads_step_b_only(self):
        draft = queue.derive()[DRAFT]
        for text in ('from step_b_gate import window_ok', 'from step_b_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,PHASES,TRIM_G4_REPEAT_S',
                     "perf=p.parent/'step-b-grade.json'", 'python3 "$H/e3_step_b_box.py" arm', 'python3 "$H/e3_step_b_box.py" grade',
                     "assert env.get('FLASH_S5_QSA_FUSED') == '1'", "assert env.get('FLASH_S5_LAYER_GROUPS') == '1'"):
            self.assertIn(text, draft)
        for text in ('from step_a_gate import', 'python3 "$H/e3_step_a_box.py"', 'q18a_step_a', 'step-a-grade', 'q18a-step-a'):
            self.assertNotIn(text, draft)
        self.assertEqual(queue.BODY.count('engine_watched '), 3)
        self.assertEqual(len(re.findall(r'^\s*python3 "\$H/', queue.BODY, flags=re.M)), 0)


class Window(unittest.TestCase):
    def test_the_box_budget_and_the_arm_entry_rule(self):
        import native40_box_gate as n40
        import native41_box_gate as n41
        self.assertEqual((gate.CLEANUP_S, gate.NOT_BEFORE, gate.HARD_END), (n40.CLEANUP_S, n40.NOT_BEFORE, n40.HARD_END))
        self.assertEqual(gate.HARD_END, 1793599200.0)      # 2026-11-02T06:00Z, inherited, not moved
        for name in n41.KEPT:
            self.assertEqual(gate.PHASES[name], n40.PHASES[name], name)
        self.assertEqual((gate.ACTIVE_S, gate.BOX_BUDGET_S, gate.LAST_ARM_ENTRY_S), (a_gate.ACTIVE_S, 9000, a_gate.LAST_ARM_ENTRY_S))
        self.assertEqual(gate.ACTIVE_S + gate.CLEANUP_S, gate.BOX_BUDGET_S)
        self.assertEqual((gate.ARMS, len(queue.ARMS), gate.BAND1_ARMS, gate.BAND2_ARMS), (16, 16, 12, 4))
        self.assertEqual(gate.ARM_RESERVE_S, 450 + 1500 + 30 + 60)
        self.assertEqual(gate.LAST_ARM_ENTRY_S + gate.ARM_RESERVE_S + gate.TAIL_RESERVE_S, gate.ACTIVE_S)
        self.assertTrue(gate.arm_fits(gate.LAST_ARM_ENTRY_S))
        self.assertFalse(gate.arm_fits(gate.LAST_ARM_ENTRY_S + 1))
        self.assertTrue(gate.g4_repeat_fits(gate.TRIM_G4_REPEAT_S))
        self.assertFalse(gate.g4_repeat_fits(gate.TRIM_G4_REPEAT_S + 1))
        self.assertLess(gate.TRIM_G4_REPEAT_S, gate.LAST_ARM_ENTRY_S)
        self.assertLess(gate.EXPECTED_RUNTIME_S['nominal'], gate.BOX_BUDGET_S)
        body = queue.BODY
        for text, count in (('timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments', 2),
                            ('timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle', 2),
                            ('timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_step_b_box.py" arm', 1),
                            ('timeout 60s python3 "$H/e3_native39_trunk_checks.py" ingress', 1),
                            ('timeout 600s python3 "$H/e3_step_b_box.py" grade', 1)):
            self.assertEqual(body.count(text), count, text)
        self.assertEqual(gate.latest_arm(), gate.HARD_END - gate.BOX_BUDGET_S)
        self.assertEqual(time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime(gate.latest_arm())), '2026-11-02T03:30Z')
        self.assertTrue(gate.window_ok(gate.latest_arm()))
        self.assertFalse(gate.window_ok(gate.latest_arm() + 1))
        self.assertFalse(gate.window_ok(gate.NOT_BEFORE - 1))
        func = tap.le_fix_function(queue.derive()[DRAFT], 'native41_budget')
        for key, value in (('active', gate.ACTIVE_S), ('cleanup', gate.CLEANUP_S), ('health', gate.PHASES['health']),
                           ('end_identity', gate.PHASES['end_identity']), ('last_arm_entry', gate.LAST_ARM_ENTRY_S),
                           ('trim_g4_repeat', gate.TRIM_G4_REPEAT_S)):
            done = subprocess.run(['bash', '-c', func + 'native41_budget ' + key],
                                  env=dict(os.environ, HARNESS_DIR=str(HERE)), capture_output=True, text=True)
            self.assertEqual((done.returncode, done.stdout.strip()), (0, str(value)), done.stderr)

    def test_draft_window_slot_executes_at_real_wall_clock(self):
        draft = queue.derive()[DRAFT]
        start = draft.index("python3 - <<'PY_SLOT'")
        slot = draft[start:draft.index('\nPY_SLOT\n', start) + len('\nPY_SLOT\n')]
        self.assertIn('step_b_gate', slot)
        done = subprocess.run(['bash', '-c', 'fail_early() { return 2; }\n' + slot],
                              env=dict(os.environ, HARNESS_DIR=str(HERE)), capture_output=True, text=True)
        expected = 0 if gate.window_ok(time.time()) else 2
        self.assertEqual(done.returncode, expected, done.stderr)
        SLOT_RECEIPTS.append(dict(draft='q18a_step_b.draft.sh', real_wall_clock=True,
                                  verdict='PASS' if done.returncode == 0 else 'OUTSIDE-WINDOW',
                                  active_s=gate.ACTIVE_S, cleanup_margin_s=gate.CLEANUP_S))


if __name__ == '__main__':
    unittest.main()
