"""Step A queue: the typed selection over the real control plane, the grade over real emitters, the arm loop in bash. CPU only.

Classes:

* Router / RouterRefused: the real attach_native_router, session, control client and driver. native_qsa_fused is forwarded
  only with FLASH_S5_QSA_FUSED=1 and FLASH_S5_AUTO_ARM=1, 409 otherwise, 400 while the engine runs.
* Startup: s5_startup.install_engine's refusals for the switch.
* Arm: the derived draft's exact arm command line through two NativeRuntimeClients over loopback HTTP into the real endpoint,
  session, control client, driver, AutoArmPolicy, NativeProgress and FusedQsa of a startup-bound stack. As in held-f's Arm,
  nothing links the HTTP prompt to the scheduler, so no native chunk runs here (a real fused chunk on a connected Core is
  e3_step_a_probe.py in the engine self-run).
* Grade: twelve arms of lines written by the real nm.ProgressTrace emitters (StepEmit), memory receipts in the engine's
  format: verdict per arm and overall, HARNESS and ENGINE, engagement both ways, memory comparison, the report's sections.
* ArmLoop: the derived body under real bash with the draft's own functions, the real death watch and the real grade, over stub
  arm and check clients.
* Commands / QueueArgparse: every python3 call of the derived texts parses with its own module's real parser.
* Derivation / Window: derived texts on disk and parse; the flags; everything else is alpha-perf line for line; the budget.

Fixture leaves: step_a_fixture_leaves.json, beside the inherited files it names.
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
import e3_step_a_box as box  # noqa: E402
import step_a_gate as gate  # noqa: E402
import step_a_queue as queue  # noqa: E402
import native_auto_arm as auto  # noqa: E402
import probe_control as pc  # noqa: E402
import s5_startup as su  # noqa: E402
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
DRAFT = 'harness/runs/q18a_step_a.draft.sh'
SETTINGS = 'harness/runs/q18a_step_a_settings.sh'
DECLARE = 'harness/runs/q18a_step_a_declare.sh'
LABELS = box.required_labels()
ARM_ACTIONS = ['status', 'pause', 'native_progress', 'native_qsa_fused', 'release_auto_arm', 'resume']
ALL = {'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst', 'FLASH_S5_QSA_FUSED': '1'}
SWITCHES = tuple(ALL)


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


@contextlib.contextmanager
def process_env(values):
    with mock.patch.dict(os.environ, values):
        for name in SWITCHES:
            if name not in values:
                os.environ.pop(name, None)
        yield


class RouterCase(hf.RouterBase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.fused = self.driver.install_qsa_fused()

    def not_selected(self):
        self.assertEqual((self.fused.selected, self.fused.counters['selections']), (False, 0))


class Router(RouterCase):
    async def test_all_settings_forward_the_control_to_the_real_session(self):
        with process_env(ALL):
            app, endpoint, _ = attach(self.frontend, ALL)
            # A running engine: the engine refuses. Its DriverError is a ValueError, which this router maps to its 400.
            self.assertEqual(await self.post(endpoint, app, 'native_qsa_fused', fused=True),
                             (400, {'error': 'invalid-command-json'}))
            self.not_selected()
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            for bad in (1, 'true', None):
                self.assertEqual(await self.post(endpoint, app, 'native_qsa_fused', fused=bad),
                                 (409, {'error': 'native-control-qsa-fused-invalid'}))
            self.not_selected()
            code, reply = await self.post(endpoint, app, 'native_qsa_fused', fused=True)
            self.assertEqual(code, 200, reply)
            self.assertEqual({k: reply[k] for k in ('action', 'fused', 'previous', 'route')},
                             dict(action='native_qsa_fused', fused=True, previous=False, route='full'))
            self.assertEqual(reply['engine_epoch'], self.driver.engine_epoch)
            self.assertEqual((self.fused.selected, self.fused.counters['selections']), (True, 1))
            code, status = await self.post(endpoint, app, 'status')
            self.assertEqual(code, 200, status)
            self.assertEqual(box.fused_view(status), dict(selected=True, run_ahead=8, stall_s=300.0, selections=1, chunks=0,
                                                          grants=0, fused_commands=0, run_ahead_polls=0, ledger_checks=0))
            # A manual control: the lease stays taken until the explicit release, and the same value counts no selection.
            self.assertIs(self.policy.manual_owned, True)
            code, reply = await self.post(endpoint, app, 'native_qsa_fused', fused=True)
            self.assertEqual((code, reply.get('previous')), (200, True))
            self.assertEqual(self.fused.counters['selections'], 1)
            code, reply = await self.post(endpoint, app, 'release_auto_arm')
            self.assertEqual((code, reply.get('released')), (200, True), reply)
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)
            self.assertEqual(await self.post(endpoint, app, 'native_qsa_fused', fused=False),
                             (400, {'error': 'invalid-command-json'}))
            self.assertIs(self.fused.selected, True)

    async def test_every_other_setting_refuses_with_409(self):
        router_off = 'native-action-not-exposed'
        session_off = 'session-action-not-exposed'
        fused_only = {'FLASH_S5_QSA_FUSED': '1'}
        auto_only = {'FLASH_S5_AUTO_ARM': '1'}
        for router, process, error in (
                (auto_only, ALL, router_off),                                      # parent's router
                (fused_only, ALL, router_off),                                     # no auto-arm: no action at all
                (dict(ALL, FLASH_S5_QSA_FUSED='0'), ALL, router_off),
                (dict(ALL, FLASH_S5_AUTO_ARM='0'), ALL, router_off),
                (ALL, auto_only, session_off),                                     # the session reads its own process
                (ALL, fused_only, session_off),
                (ALL, dict(ALL, FLASH_S5_QSA_FUSED='0'), session_off),
                (ALL, dict(ALL, FLASH_S5_AUTO_ARM='0'), session_off)):
            with self.subTest(router=router, process=process), process_env(process):
                app, endpoint, _ = attach(self.frontend, router)
                self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
                for fused in (True, False):
                    self.assertEqual(await self.post(endpoint, app, 'native_qsa_fused', fused=fused),
                                     (409, {'error': error}))
                self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)
                self.not_selected()
                self.assertIs(self.policy.manual_owned, True)


class RouterNotInstalled(hf.RouterBase):
    async def test_an_engine_without_the_switch_refuses_and_its_status_has_no_block(self):
        with process_env(ALL):
            app, endpoint, _ = attach(self.frontend, ALL)
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            self.assertEqual(await self.post(endpoint, app, 'native_qsa_fused', fused=True),
                             (400, {'error': 'invalid-command-json'}))
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)
            code, status = await self.post(endpoint, app, 'status')
            self.assertEqual(code, 200)
            self.assertNotIn('native_qsa_fused', status)
            self.assertIsNone(box.fused_view(status))


class Startup(unittest.TestCase):
    def refuses(self, env, text):
        env = dict(env, FLASH_S5_OVERLAP='1', FLASH_S5_NATIVE_READY='1')
        with self.assertRaisesRegex(su.StartupError, text):
            su.install_engine(SimpleNamespace(), env=env)

    def test_the_switch_needs_auto_arm_and_the_dense_policy(self):
        needs = 'FLASH_S5_QSA_FUSED needs FLASH_S5_AUTO_ARM=1 and the dense QSA policy'
        dense = {'FLASH_S5_NATIVE_QSA_POLICY': 'dense'}
        self.refuses(dict(dense, FLASH_S5_QSA_FUSED='1'), needs)
        self.refuses(dict(dense, FLASH_S5_QSA_FUSED='1', FLASH_S5_AUTO_ARM='0'), needs)
        self.refuses({'FLASH_S5_QSA_FUSED': '1', 'FLASH_S5_AUTO_ARM': '1'}, needs)
        for bad in ('true', '2', ''):
            self.refuses({'FLASH_S5_QSA_FUSED': bad}, 'invalid boolean setting: FLASH_S5_QSA_FUSED')


# --- arms over the real control plane -----------------------------------------------------------------

def arm_argv(**values):
    (args,) = [a for m, a in body_commands(queue.derive()[DRAFT]) if m == 'e3_step_a_box' and a.startswith('arm ')]
    values = dict(tap.QUEUE_VALUES, **values)
    values['_label'] = '%s-%s-%s' % (values['_mode'], values['_cohort'], values['_pair'])
    return held.expand(args, values)


def body_commands(text):
    body = text[text.index(queue.BODY.splitlines()[0]):]
    body = body[:body.index('record step-a ')]
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
        self.stack, self.core = native38.OwnerStartup().stack()
        held.pinned_reads(self.stack.scheduler)
        self.ready = native38.bootstrap_full_startup(self.core, self.stack.driver, timeout_s=10)
        self.driver = self.stack.driver
        # s5_startup.install_engine's order: the auto-arm policy, progress, the fused switch.
        self.policy = auto.install(self.core, self.driver, {auto.ENV: '1'})
        self.progress = self.driver.install_progress('burst')
        self.fused = self.driver.install_qsa_fused() if self.INSTALL else None
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

    def test_both_selections_in_both_bands_through_the_exact_cli(self):
        salts = {}
        before = progress.label, progress.release, progress.progress_reasons, progress.progress_view, progress.ACTIONS
        actions_before = progress.base.ARM_ACTIONS
        for selection, cohort, index, name, budget, band, previous in (
                ('fused', 'live', 2, 'q18a-alpha-perf-band2', 32768, 2, False),
                ('stepped', 'live', 2, 'q18a-alpha-perf-band2', 32768, 2, True),
                ('stepped', 'solo', 0, 'q18a-native41-progress', 16384, 1, False),
                ('fused', 'solo', 1, 'q18a-native41-progress', 16384, 1, False)):
            with self.subTest(selection=selection, cohort=cohort, index=index):
                self.latch()
                rc, record, actions, served = self.run_arm(selection, cohort, index, name)
                self.assertEqual(rc, 2)      # No native chunk here, by construction: incomplete by its own receipts.
                self.assertNotIn('error', record, record.get('error'))
                self.assertEqual((record['label'], record['selection'], record['mode'], record['band'], record['workload']),
                                 (box.label(selection, cohort, index), selection, 'burst', band, name))
                self.assertEqual(actions[:6], ARM_ACTIONS)
                self.assertEqual(set(actions), set(ARM_ACTIONS))
                self.assertEqual([actions.count(a) for a in ARM_ACTIONS[1:]], [1, 1, 1, 1, 1])
                self.assertEqual(record['selection_switch']['fused'], selection == 'fused')
                self.assertEqual((record['selection_switch']['previous'], record['selection_switch']['route']), (previous, 'full'))
                self.assertEqual((record['mode_switch']['mode'], self.progress.mode), ('burst', 'burst'))
                self.assertEqual((self.fused.selected, record['release']['released']), (selection == 'fused', True))
                self.assertEqual({r['selected'] for r in (s['native_qsa_fused'] for s in record['progress']
                                                          if s['label'] != 'entry')}, {selection == 'fused'})
                self.assertEqual(record['fused_delta'], dict.fromkeys(box.FUSED_COUNTERS, 0))
                self.assertEqual(record['fused_delta_append'], dict.fromkeys(box.FUSED_COUNTERS, 0))
                self.assertEqual([c['max_tokens'] for c in served], [budget] * 8 + [1, 1])
                self.assertIn('progress-chunks-not-recorded', record['reasons'])
                self.assertNotIn('control-sequence-not-pause-native-progress-release-resume', record['reasons'])
                self.assertNotIn('arm-sent-a-non-receipt-action', record['reasons'])
                self.assertNotIn('control-sequence-not-pause-progress-selection-release-resume', record['reasons'])
                self.assertFalse([x for x in record['reasons'] if x.startswith('auto-not-proven:unexpected')], record['reasons'])
                salts[record['label']] = record['salt_sha256']
        self.assertEqual(len(set(salts.values())), 4)
        # The seams are this process's one arm; they are back after each.
        self.assertEqual((progress.label, progress.release, progress.progress_reasons, progress.progress_view, progress.ACTIONS),
                         before)
        self.assertIs(progress.base.ARM_ACTIONS, actions_before)

    def test_the_wrong_manifest_for_the_round_is_refused_before_any_traffic(self):
        rc, record, actions, served = self.run_arm('fused', 'live', 2, 'q18a-native41-progress')
        self.assertEqual((rc, record['reasons'], record['label']), (2, ['arm-refused'], 'fused-live-2'))
        self.assertEqual(record['error']['error_message'], 'alpha-perf-workload-declaration-invalid')
        self.assertEqual((actions, served), ([], []))
        self.assertFalse(self.fused.selected)

    def test_rounds_and_selections_outside_the_declared_set_do_not_parse(self):
        for selection, index in (('fused', '3'), ('burst', '0'), ('current', '0')):
            argv = arm_argv(_mode=selection, _cohort='live', _pair=index, Q=self.tmp.name)
            with self.subTest(selection=selection), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                box.main(argv)
            self.assertFalse(Path(self.tmp.name, 'arm-%s-live-%s.json' % (selection, index)).exists())


class ArmNotInstalled(Arm):
    INSTALL = False

    def test_both_selections_in_both_bands_through_the_exact_cli(self):
        self.latch()
        rc, record, actions, served = self.run_arm('fused', 'solo', 0, 'q18a-native41-progress')
        self.assertEqual((rc, record['complete'], served), (2, False, []))
        self.assertIn('arm-error', record['reasons'])
        self.assertEqual(record['error']['error_message'], 'native-qsa-fused-not-installed')
        self.assertEqual(actions, ['status'])

    def test_the_wrong_manifest_for_the_round_is_refused_before_any_traffic(self):
        pass

    def test_rounds_and_selections_outside_the_declared_set_do_not_parse(self):
        pass


class Selection(unittest.TestCase):
    """The arm's own proofs over declared client replies (leaf arm-control-replies), each defect its row."""

    class Client:
        def __init__(self, fused_reply=None, route='full', released=True, raises=None):
            self.fused_reply, self.route, self.released, self.raises = fused_reply, route, released, raises
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
            if action == 'native_qsa_fused':
                fused = fields['fused'] if self.fused_reply is None else self.fused_reply
                return dict(fused=fused, previous=False, route=self.route, route_generation=1, engine_epoch=7)
            return dict(released=self.released, engine_epoch=7)

    def test_order_and_the_reply_is_checked(self):
        client, record = self.Client(), {}
        box.release_fused(True)(client, record, 'burst')
        self.assertEqual(client.calls, ['pause', 'native_progress', 'native_qsa_fused', 'release_auto_arm', 'resume'])
        for client, text in ((self.Client(fused_reply=False), 'native-qsa-fused-selection-unproved'),
                             (self.Client(route='split'), 'native-qsa-fused-selection-unproved'),
                             (self.Client(released=False), 'auto-arm-release-unproved')):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, text):
                box.release_fused(True)(client, {}, 'burst')
            self.assertEqual(client.calls[-1], 'resume')

    def test_a_refused_selection_still_resumes_and_never_releases(self):
        client = self.Client(raises='native_qsa_fused')
        with self.assertRaisesRegex(ValueError, 'refused-by-the-engine'):
            box.release_fused(True)(client, {}, 'burst')
        self.assertEqual(client.calls, ['pause', 'native_progress', 'native_qsa_fused', 'resume'])

    def receipts(self, *, history=None, before=None, after=None):
        block = dict(selected=True, selections=1, chunks=0, grants=0, fused_commands=0, run_ahead_polls=0, ledger_checks=0)
        record = dict(control_history=['status'] + ARM_ACTIONS[1:] if history is None else history, progress=[
            dict(label='before-fresh', native_qsa_fused=dict(block, **(before or {}))),
            dict(label='after-fresh', native_qsa_fused=dict(block, chunks=3, **(after or {})))])
        return box.selection_reasons(record, True), record

    def test_selection_receipts(self):
        reasons, record = self.receipts()
        self.assertEqual((reasons, record['fused_delta']['chunks']), ([], 3))
        self.assertIn('fused-selection-not-this-arm', self.receipts(before=dict(selected=False))[0])
        self.assertIn('fused-selection-not-this-arm', self.receipts(after=dict(selected=False))[0])
        self.assertIn('fused-selection-moved-during-the-fresh-prompt', self.receipts(after=dict(selections=2))[0])
        order = ['status', 'pause', 'native_qsa_fused', 'native_progress', 'release_auto_arm', 'resume']
        self.assertIn('control-sequence-not-pause-progress-selection-release-resume', self.receipts(history=order)[0])
        twice = ARM_ACTIONS[:4] + ['native_qsa_fused'] + ARM_ACTIONS[4:]
        self.assertIn('arm-sent-the-selection-more-than-once', self.receipts(history=twice)[0])


# --- grade --------------------------------------------------------------------------------------------

GPU = {('fused', 1): (21000, 22000, 900), ('stepped', 1): (22000, 23000, 400),
       ('fused', 2): (21500, 22500, 500), ('stepped', 2): (22500, 23300, 40)}


class StepEmit(hf.Emit):
    """hf.Emit plus the two protocols' command rows (the real row layout, real emitters).

    Stepped: twelve profile rows per chunk, as every chunk of held-f run 2's engine stream. Fused: the layer segments alone.
    """

    def chunk(self, start, end, *, selection, live, layers=12):
        self.txn += 1
        kinds = [('open', None)]
        for _ in range(layers):
            if selection == 'stepped':
                kinds.append(('profile', None))
            kinds.append(('segment', 'layer'))
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
        allocated, reserved, free = GPU[selection, band]
        for point, retries, shift in (('open', 5 + txn, 0), ('released', 6 + txn, 1)):
            self.log(self.now, rank, 'flash-s5 memory point=%s rank=%d txn_seq=%d chunk=%d-%d allocated=%d reserved=%d '
                                     'inactive_split=%d alloc_retries=%d ooms=0 free=%d total=25296044032' % (
                                         point, rank, txn, start, end, (allocated + shift) * 2 ** 20, reserved * 2 ** 20,
                                         2 ** 26, retries, (free - shift) * 2 ** 20))


def synthetic_run(order=None, extra=None, *, rate=None):
    """The declared arms in order, each arm's real emitter lines, memory receipts and fused counters."""
    emit, arms, extra = StepEmit(), [], extra or {}
    for selection, cohort, index in (order or box.ARM_ORDER):
        name, band = box.label(selection, cohort, index), ap.band_of_index(index)
        cuts = tap.band_cuts(band)
        start = emit.tick(5_000_000)
        emit.tick(500_000)
        chunks = len(cuts) - 1
        lives = [8] * chunks if cohort == 'live' else [8] + [0] * (chunks - 1)
        for c, live in enumerate(lives):
            for rank in range(4):
                emit.memory(selection, band, rank, c + 1, cuts[c], cuts[c + 1])
            emit.chunk(cuts[c], cuts[c + 1], selection=selection, live=live)
        if name in extra:
            emit.log(emit.now, 1, extra[name])
        end = emit.tick(1_000_000)
        streams = [40.0, 41.0, 42.0, 42.0, 43.0, 43.0, 44.0, 45.0]
        window = {'aggregate_tokens_per_s': 340.0, 'stream_rates_tokens_per_s': streams, 'gap_ms_p99': 60.0, 'gap_ms_max': 100.0}
        client = {'prefill_tokens_per_s': (rate or {}).get(selection, 900.0), 'all_open_share_of_span': 1.0 if cohort == 'live' else 0.1}
        if cohort == 'live':
            client['decode_window'] = window
        fused = selection == 'fused'
        delta = dict(selections=0, chunks=chunks if fused else 0, grants=12 * chunks if fused else 0,
                     fused_commands=12 * chunks if fused else 0, run_ahead_polls=0, ledger_checks=50 * chunks if fused else 0)
        arms.append(dict(label=name, selection=selection, mode='burst', cohort=cohort, index=index, band=band, complete=True,
                         reasons=[], wall_start_s=emit.wall(start), wall_end_s=emit.wall(end),
                         progress_delta={'chunks': chunks, 'commands': chunks * 26}, fused_delta=delta,
                         selection_switch=dict(fused=fused, route='full'),
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

    def test_twelve_arms_pass_with_all_sections(self):
        emit, arms = synthetic_run()
        result = self.grade(emit, arms)
        self.assertEqual((result['verdict'], result['missing'], result['failures'], result['classification']),
                         ('PASS', [], [], None), result['missing'])
        self.assertEqual(result['required_arms'], LABELS)
        self.assertEqual({v['verdict'] for v in result['validity'].values()}, {'PASS'})
        row = result['table']['fused-live-2']
        self.assertEqual((row['band'], row['selection'], row['mode'], row['cohort'], row['prompt_tokens'], row['chunks']),
                         (2, 'fused', 'burst', 'live', 131094, 5))
        self.assertEqual((row['ttft_s'], row['s_per_chunk'], row['prefill_tok_s']), (5.0, 1.0, 900.0))
        self.assertEqual((row['min_stream_rate'], row['gap_p99_ms'], row['gap_max_ms'], row['commands_per_chunk']),
                         (40.0, 60.0, 100.0, 26.0))
        self.assertAlmostEqual(row['p10_stream_rate'], 40.7)
        self.assertAlmostEqual(row['p10_stream_ratio'], 40.7 / 50)
        self.assertEqual(row['memory']['worst'], dict(
            peak_allocated_mib=21501.0, peak_reserved_mib=22500.0, min_free_mib=499.0, alloc_retries=4 * 5, ooms=0, receipts=40))
        self.assertIn('97152-114688', row['chunk_breakdown'])
        engaged = result['validity']['fused-live-2']['engagement']
        self.assertEqual((engaged['engaged'], engaged['owned_chunks'], engaged['profile_max']), (True, 5, 0))
        engaged = result['validity']['stepped-live-2']['engagement']
        self.assertEqual((engaged['engaged'], engaged['profile_min'], engaged['profile_max']), (True, 12, 12))
        band2 = result['summary']['band2']
        self.assertEqual(sorted(band2['groups']), ['fused-live', 'fused-solo', 'stepped-live', 'stepped-solo'])
        self.assertEqual(band2['groups']['fused-live']['arms'], ['fused-live-2'])
        self.assertEqual(band2['fused_over_stepped']['live']['prefill_tok_s'], 1.0)
        pair = result['memory_fused_vs_stepped']['live-2']
        self.assertEqual((pair['fused_minus_stepped']['peak_reserved_mib'], pair['fused_minus_stepped']['min_free_mib']),
                         (-800.0, 460.0))
        self.assertEqual(sorted(result['memory_fused_vs_stepped']), ['live-0', 'live-1', 'live-2', 'solo-0', 'solo-1', 'solo-2'])
        self.assertEqual((result['speed_is_gate'], result['memory_is_gate']), (False, False))
        self.assertIn('Not assessed by this queue', result['release_readiness'])
        self.assertIn('no lifecycle, numerics or leak evidence', result['release_readiness'])

    def test_speed_and_memory_never_change_the_verdict(self):
        emit, arms = synthetic_run(rate=dict(fused=1.0))
        self.assertEqual(self.grade(emit, arms)['verdict'], 'PASS')

    def test_a_fused_arm_that_did_not_engage_is_inc(self):
        emit, arms = synthetic_run()
        base_delta = next(a['fused_delta'] for a in arms if a['label'] == 'fused-live-0')
        for field, value, reason in (('fused_commands', 12 * 3 - 1, 'fused-commands-below-twelve-per-chunk'),
                                     ('chunks', 2, 'fused-chunks-delta-differs-from-the-auto-owned-chunks'),
                                     ('chunks', 0, 'fused-chunks-delta-differs-from-the-auto-owned-chunks')):
            with self.subTest(field=field, value=value):
                result = self.grade(emit, self.edited(arms, 'fused-live-0', fused_delta=dict(base_delta, **{field: value})))
                self.assertEqual((result['verdict'], result['failures']), ('INC', []))
                self.assertIn('fused-live-0:not-engaged', result['missing'])
                row = result['validity']['fused-live-0']
                self.assertEqual((row['verdict'], row['classification'], row['reasons']), ('INC', None, [reason]))
        # Exactly the bound passes; more is allowed (a layer can take more than one command).
        self.assertEqual(result['validity']['fused-live-2']['verdict'], 'PASS')
        result = self.grade(emit, self.edited(arms, 'fused-live-0', fused_delta=dict(base_delta, fused_commands=99)))
        self.assertEqual(result['verdict'], 'PASS')
        # The receipts themselves missing.
        result = self.grade(emit, self.edited(arms, 'fused-live-0', fused_delta=None))
        self.assertEqual(result['validity']['fused-live-0']['reasons'], ['engagement-receipts-missing'])
        self.assertEqual(result['verdict'], 'INC')

    def test_a_fused_arm_running_the_stepped_protocol_is_inc_and_the_reverse(self):
        # Lines of the other protocol under the receipts of this one.
        emit, arms = synthetic_run()
        order = [(('stepped' if s == 'fused' else 'fused'), c, i) for s, c, i in box.ARM_ORDER]
        other_emit, other = synthetic_run(order)
        relabel = {box.label(*old): box.label(*new) for old, new in zip(order, box.ARM_ORDER)}
        swapped = [dict(a, label=relabel[a['label']], selection=next(x for x in arms if x['label'] == relabel[a['label']])['selection'],
                        fused_delta=next(x for x in arms if x['label'] == relabel[a['label']])['fused_delta']) for a in other]
        result = self.grade(other_emit, swapped)
        self.assertEqual(result['verdict'], 'INC')
        for selection, reason in (('fused', 'fused-arm-sent-profile-queries'), ('stepped', 'stepped-arm-without-the-stepped-protocol')):
            self.assertIn(reason, result['validity'][selection + '-solo-0']['reasons'])

    def test_a_stepped_arm_that_moved_a_fused_counter_is_inc(self):
        emit, arms = synthetic_run()
        for field in box.FUSED_COUNTERS:
            with self.subTest(field=field):
                delta = dict.fromkeys(box.FUSED_COUNTERS, 0)
                delta[field] = 1
                result = self.grade(emit, self.edited(arms, 'stepped-solo-1', fused_delta=delta))
                self.assertEqual(result['validity']['stepped-solo-1']['reasons'], ['stepped-arm-moved-a-fused-counter'])
                self.assertEqual(result['verdict'], 'INC')

    def test_each_engine_signature_fails_that_arm_as_engine(self):
        for line, name in (('(EngineCore_DP0 pid=101) ERROR native driver failed: tp-ledger-mismatch stage=0', 'tp-ledger-mismatch'),
                           ('(Worker_TP1 pid=202) ERROR raise DriverError(tp-collective-stall)', 'tp-collective-stall'),
                           ('(Worker_TP1 pid=202) ERROR tp_collective_stall after 300 s', 'tp-collective-stall'),
                           ('(EngineCore_DP0 pid=101) ERROR fused-profile-differs-from-the-admission', 'fused-profile'),
                           ('(Worker_TP1 pid=202) ERROR QSA layers disagree on the packed selection width', 'tp-disagree')):
            with self.subTest(name=name):
                emit, arms = synthetic_run(extra={'fused-live-2': line})
                result = self.grade(emit, arms)
                row = result['validity']['fused-live-2']
                self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))
                self.assertEqual((row['verdict'], row['classification']), ('FAIL', 'ENGINE'))
                self.assertIn(name, row['reasons'])
                self.assertEqual({v['verdict'] for k, v in result['validity'].items() if k != 'fused-live-2'}, {'PASS'})
                self.assertEqual(result['summary']['band2']['arm_verdicts']['fused-live-2'], 'FAIL')

    def test_a_signature_between_the_arm_spans_fails_the_run(self):
        emit, arms = synthetic_run()
        gap = arms[3]['wall_end_s'] + progress.LOG_LAG_S + 0.5
        emit.log(emit.mono0 + int(round((gap - hf.WALL0) * 1e9)), 1, 'ERROR tp-ledger-mismatch between arms')
        result = self.grade(emit, arms)
        self.assertEqual((result['verdict'], result['failures'], result['engine_lines_outside_every_arm']),
                         ('FAIL', ['engine-signature-outside-every-arm-span'], 1))
        self.assertEqual({v['verdict'] for v in result['validity'].values()}, {'PASS'})

    def test_a_signature_inside_a_progress_row_is_no_signature(self):
        self.assertIsNone(box.signature_of(progress.PROGRESS_LINE + 'tp-ledger-mismatch'))
        self.assertEqual(box.signature_of('x fused-profile-disagreement y'), 'fused-profile')

    def test_the_error_of_an_arm_counts_its_signature(self):
        emit, arms = synthetic_run()
        changed = self.edited(arms, 'fused-solo-0', complete=False, reasons=['arm-error'],
                              error={'error_message': 'native-response: tp-collective-stall on stage 0'})
        result = self.grade(emit, changed)
        row = result['validity']['fused-solo-0']
        self.assertEqual((row['verdict'], row['classification'], row['signatures']), ('FAIL', 'ENGINE', ['tp-collective-stall']))

    def test_engine_watch_ingress_and_incomplete_rows(self):
        emit, arms = synthetic_run()
        name = 'stepped-live-0'
        watches = {a['label']: tap.ALIVE for a in arms}
        result = self.grade(emit, arms, watches=dict(watches, **{name: {'verdict': 'DEAD'}}))
        self.assertEqual((result['verdict'], result['classification'], result['validity'][name]['crash_free']),
                         ('FAIL', ['ENGINE'], False))
        # A watch stopped the client: no arm record, the row is explicit.
        kept = [a for a in arms if a['label'] != name]
        result = self.grade(emit, kept, watches=dict(watches, **{name: {'verdict': 'DEAD'}}))
        self.assertEqual(result['validity'][name]['reasons'], ['engine-dead'])
        self.assertEqual(result['validity'][name]['classification'], 'ENGINE')
        # An arm never entered is INC, never PASS.
        watches.pop(name)
        result = self.grade(emit, kept, watches=watches)
        self.assertEqual((result['verdict'], result['validity'][name]['verdict']), ('INC', 'INC'))
        self.assertIn('declared-arm-missing:' + name, result['missing'])
        # A missing watch record is not a pass; nor is an incomplete arm; a failed ingress is the harness.
        result = self.grade(emit, arms, watches=watches)
        self.assertEqual((result['verdict'], result['validity'][name]['reasons']), ('INC', ['engine-watch-record-missing']))
        result = self.grade(emit, self.edited(arms, name, complete=False, reasons=['fresh-not-complete']))
        self.assertEqual((result['verdict'], result['validity'][name]['verdict']), ('INC', 'INC'))
        result = self.grade(emit, arms, ingress=dict({a['label']: tap.PASS for a in arms}, **{name: {'verdict': 'FAIL'}}))
        self.assertEqual((result['verdict'], result['classification'], result['validity'][name]['classification']),
                         ('FAIL', ['HARNESS'], 'HARNESS'))

    def test_held_f_completeness_defects_hold(self):
        emit, arms = synthetic_run()
        span = next(a for a in arms if a['label'] == 'fused-solo-2')
        inside = [i for i, (s, t) in enumerate(emit.lines) if s == 'err' and progress.PROGRESS_LINE in t
                  and span['wall_start_s'] <= progress.docker_stamp(t) <= span['wall_end_s']]
        lines = [x for i, x in enumerate(emit.lines) if i != inside[1]]
        result = self.grade(emit, arms, lines=lines)
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('progress-fused-solo-2:worker-rows-missing', result['missing'])
        self.assertIn('progress:worker-rows-missing', result['validity']['fused-solo-2']['reasons'])
        changed = self.edited(arms, 'fused-live-2', band=1)
        self.assertIn('fused-live-2:band-not-the-round-index-band', self.grade(emit, changed)['missing'])
        result = self.grade(emit, self.edited(arms, 'fused-live-2', progress_delta={'chunks': 6}))
        self.assertIn('progress-fused-live-2:progress-lines-differ-from-the-chunk-counter', result['missing'])
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

    def test_the_report_keeps_three_separate_sections(self):
        emit, arms = synthetic_run()
        result = self.grade(emit, arms)
        text = box.report(result)
        heads = re.findall(r'^## (.+)$', text, flags=re.M)
        self.assertEqual(heads, ['Mechanism', 'Performance', 'Release readiness'])
        self.assertEqual(text.split('## Release readiness\n\n')[1].strip(), box.RELEASE_READINESS)
        mechanism = text[text.index('## Mechanism'):text.index('## Performance')]
        for name in LABELS:
            self.assertIn('| %s | PASS |' % name, mechanism)
        performance = text[text.index('## Performance'):text.index('## Release readiness')]
        for field in ('### Band 1', '### Band 2', 'Fused over stepped', 'Device memory', 'TTFT', 'p10 stream ratio'):
            self.assertIn(field, performance)
        self.assertNotIn('PASS', performance)
        # An inconclusive run still prints, with a missing row and a missing memory pair.
        kept = [a for a in arms if a['label'] != 'fused-solo-2']
        text = box.report(self.grade(emit, kept))
        self.assertIn('Overall: **INC**', text)
        self.assertIn('| solo-2 | not available |', text)

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
        self.assertEqual(out.getvalue().strip(), 'STEP_A_GRADE PASS')
        self.assertEqual(json.loads((q / 'grade.json').read_text())['schema'], box.GRADE_SCHEMA)
        self.assertTrue((q / 'r.md').read_text().startswith('# q18a-step-a report'))
        self.assertEqual(box.main(['grade'] + args + ['--out', str(q / 'grade.json')]), 2)
        # An unreadable arm record stays a row; the rc is the verdict's.
        (q / 'arm-fused-solo-2.json').write_text('{not json')
        args[args.index(str(q / 'r.md'))] = str(q / 'r2.md')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(box.main(['grade'] + args + ['--out', str(q / 'grade-2.json')]), 2)
        self.assertIn('declared-arm-missing:fused-solo-2', json.loads((q / 'grade-2.json').read_text())['missing'])


# --- the derived arm loop under real bash ------------------------------------------------------------

STUB_BOX = tap.STUB_BOX.replace('e3_alpha_perf_box', 'e3_step_a_box')


class ArmLoop(unittest.TestCase):
    """queue.BODY runs as written; arm and check clients are stubs (leaf arm-loop-clients), the watch and grade real."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='step-a-loop-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q, self.h, self.bin, self.arms = (self.root / x for x in ('q', 'h', 'bin', 'arms'))
        for d in (self.q, self.h, self.bin, self.arms):
            d.mkdir()
        for name in ('native43_engine_death.py', 'step_a_gate.py', 'native39_trunk_gate.py', 'native40_box_gate.py',
                     'native41_box_gate.py'):
            (self.h / name).symlink_to(HERE / name)
        (self.h / 'e3_native39_trunk_checks.py').write_text(tap.STUB_CHECKS)
        (self.h / 'e3_step_a_box.py').write_text(STUB_BOX)
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
        return json.loads((self.q / 'step-a-grade.json').read_text())

    def test_twelve_arms_in_order_with_the_band_manifest_and_env_then_the_real_grade(self):
        out, err, calls, steps = self.run_loop(None)
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, (out[-1500:], err[-800:]))
        arm_calls = [c for c in calls if c['script'] == 'box']
        self.assertEqual([c['label'] for c in arm_calls], LABELS)
        for c in arm_calls:
            expected = 'q18a-native41-progress.json' if c['label'][-1] in '01' else 'q18a-alpha-perf-band2.json'
            self.assertEqual(c['workload'], expected, c['label'])
            self.assertEqual(c['env'], dict(READINESS='/r/ready', GEN='/o/S5-OVERLAP-GENERATION.json', S5_OVERLAY='/o',
                                            S5_LAUNCHER_NAME='l.sh', PACKAGE_MANIFEST='/o/SHA256SUMS',
                                            GENERATION='/o/S5-OVERLAP-GENERATION.json'))
        self.assertEqual([c['out'] for c in calls if c['script'] == 'checks' and c['action'] == 'full-idle'],
                         ['entry-%s.json' % x for x in LABELS] + ['end-full-idle.json'])
        for name in LABELS:
            self.assertEqual(steps['arm-' + name], 'PASS')
            watch = json.loads((self.q / ('engine-watch-arm-%s.json' % name)).read_text())
            self.assertEqual((watch['verdict'], watch['client_rc'], watch['container']), ('CLIENT_EXIT', 0, 'qwen-flash-prod'))
        self.assertEqual(steps['step-a'], 'PASS')
        result = self.grade_file()
        self.assertEqual((result['verdict'], result['missing']), ('PASS', []))
        self.assertEqual({v['engine_watch'] for v in result['validity'].values()}, {'CLIENT_EXIT'})
        self.assertEqual(re.findall(r'^## (.+)$', (self.q / 'step-a-report.md').read_text(), flags=re.M),
                         ['Mechanism', 'Performance', 'Release readiness'])

    def test_budget_before_an_arm_is_not_run_and_inconclusive(self):
        out, err, calls, steps = self.run_loop(box.ARM_ORDER[:9], jump_after='ingress-fused-live-2')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box'], LABELS[:9])
        for name in LABELS[9:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
            self.assertFalse((self.q / ('entry-%s.json' % name)).exists())
        self.assertIn('END FAILS=0 INCS=4 CRASHED=0', out, out[-800:])       # the three arms and the grade
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('declared-arm-missing:stepped-live-2', result['missing'])

    def test_a_dead_engine_stops_the_band_2_arm_and_keeps_band_1(self):
        started = time.monotonic()
        out, err, calls, steps = self.run_loop(box.ARM_ORDER[:8], die_in='fused-live-2')
        self.assertLess(time.monotonic() - started, 120)
        self.assertIn('CRASHED=1 SOURCE=engine-death', out, out[-800:])
        self.assertEqual(steps['arm-fused-live-2'], 'FAIL')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box'], LABELS[:9])
        for name in LABELS[9:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
        watch = json.loads((self.q / 'engine-watch-arm-fused-live-2.json').read_text())
        self.assertEqual(watch['verdict'], 'DEAD')
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('fused-live-2:engine-dead', result['failures'])
        self.assertEqual((result['validity']['fused-live-2']['verdict'], result['validity']['fused-live-2']['classification']),
                         ('FAIL', 'ENGINE'))
        self.assertEqual({result['validity'][x]['verdict'] for x in LABELS[:8]}, {'PASS'})
        self.assertEqual(sorted(result['summary']['band1']['arm_verdicts']), sorted(LABELS[:8]))


# --- commands, derivation, window ----------------------------------------------------------------------

ARGPARSE_VALUES = dict(tap.ARGPARSE_VALUES, QNAME='q18a-step-a', RUN_ID='q18a-step-a-fixture', _mode='fused', _cohort='live',
                       _pair='2', _label='fused-live-2')


def bash_words(args, draft, values=None):
    budget = tap.le_fix_function(draft, 'native41_budget')
    values = dict(ARGPARSE_VALUES, **(values or {}))
    import shlex
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(HERE))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    lines += ['GRADE_ARGS=(--arm "$Q/arm-fused-live-2.json" --ingress "fused-live-2=$Q/ingress-fused-live-2.json" '
              '--watch "fused-live-2=$Q/engine-watch-arm-fused-live-2.json")',
              'source "$H/runs/q18a_step_a_settings.sh" || exit 97', budget, "printf '%s\\0' " + args]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode().split('\0')[:-1], done.stderr.decode()[-400:]


class Commands(unittest.TestCase):
    def test_every_body_command_parses_with_its_module(self):
        import e3_native39_trunk_checks as checks
        modules = {'e3_step_a_box': box, 'e3_native39_trunk_checks': checks}
        rows = body_commands(queue.derive()[DRAFT])
        self.assertEqual(sorted({m for m, _ in rows}), sorted(modules))
        actions = []
        original = argparse.ArgumentParser.parse_args

        def capture(parser, args=None, namespace=None):
            raise held.Parsed(original(parser, args, namespace))
        for module, args in rows:
            argv = held.expand(args, dict(tap.QUEUE_VALUES, _mode='fused', _cohort='live', _pair='2', _label='fused-live-2',
                                          _workload='/h/workloads/q18a-alpha-perf-band2.json'))
            with mock.patch.object(argparse.ArgumentParser, 'parse_args', capture):
                with self.assertRaises(held.Parsed) as parsed, contextlib.redirect_stderr(io.StringIO()):
                    modules[module].main(argv)
            actions.append((module, argv[0], vars(parsed.exception.args[0])))
        self.assertEqual([a[1] for a in actions], ['instruments', 'full-idle', 'arm', 'ingress', 'instruments', 'full-idle', 'grade'])
        arm = next(ns for m, a, ns in actions if a == 'arm')
        self.assertEqual((arm['mode'], arm['cohort'], arm['index'], arm['control_timeout_s']), ('fused', 'live', 2, 60.0))
        grade = actions[-1][2]
        self.assertEqual((grade['out'], grade['report']), ('/q/step-a-grade.json', '/q/step-a-report.md'))
        self.assertEqual(grade['log'], ['/q/engine-stream.log', '/q/engine-stream.err'])
        ingress = next(ns for m, a, ns in actions if a == 'ingress')
        self.assertEqual(ingress['generation'], progress.base.ARM_GENERATIONS)

    def test_the_loop_list_is_the_lead_order_and_the_split_runs_in_bash(self):
        draft = queue.derive()[DRAFT]
        (loop,) = re.findall(r'^\s*for _label in ([a-z0-9 -]+); do$', draft, flags=re.M)
        self.assertEqual(loop.split(), LABELS)
        self.assertEqual(list(queue.ARMS), LABELS)
        self.assertEqual(queue.ARMS_BAND1, ('stepped-solo-0', 'fused-solo-0', 'fused-live-0', 'stepped-live-0',
                                            'stepped-live-1', 'fused-live-1', 'fused-solo-1', 'stepped-solo-1'))
        self.assertEqual(queue.ARMS_BAND2, ('fused-live-2', 'stepped-live-2', 'stepped-solo-2', 'fused-solo-2'))
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
            'e3_native39_trunk_checks.py', 'e3_native39_w2_length.py', 'e3_step_a_box.py',
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
        arm, grade = first('e3_step_a_box.py', 'arm'), first('e3_step_a_box.py', 'grade')
        for label, original, (old, new) in (('arm-unknown-option', arm, ('--control-timeout-s', '--control-timeout-sec')),
                                            ('arm-undeclared-mode', arm, ('--mode "$_mode"', '--mode burst')),
                                            ('arm-undeclared-round', arm, ('--index "$_pair"', '--index 6')),
                                            ('grade-without-out', grade, ('--out "$Q/step-a-grade.json"', ''))):
            with self.subTest(control=label):
                self.assertEqual(original.count(old), 1)
                rc, words, err = bash_words(original.replace(old, new), draft)
                self.assertEqual(rc, 0, err)
                self.assertEqual(held.real_parse(HERE / 'e3_step_a_box.py', words)[0], 'refused', label)


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
                     'FLASH_S5_QSA_FUSED=1', 'FLASH_S5_REPLAY_OBSERVER=0', 'FLASH_S5_ENGINE_OFFERS=0'):
            self.assertEqual(words.count(flag), 1, flag)
        for absent in ('FLASH_S5_UNION_RETIRE', 'FLASH_S5_NATIVE_QSA_TRACE', 'FLASH_S5_NATIVE_QSA_STOP_AFTER'):
            self.assertNotIn(absent, extra[0])

    def test_settings_add_that_token_and_its_comment_nothing_else(self):
        old = self.rename_old((ROOT / 'harness/runs/q18a_alpha_perf_settings.sh').read_text())
        new = queue.derive()[SETTINGS]
        diff = [line for line in difflib.ndiff(old.splitlines(), new.splitlines()) if line[:1] in '+-']
        self.assertEqual([line[2:].replace(' ' + queue.FUSED, '') for line in diff if line[0] == '+' and 'EXTRA_ENV' in line],
                         [line[2:] for line in diff if line[0] == '-'])
        self.assertEqual([line[2:] for line in diff if line[0] == '+' and 'EXTRA_ENV' not in line], [queue.SETTINGS_NOTE.strip()])

    def test_the_declaration_adds_one_assert_and_the_pins(self):
        old = self.rename_old((ROOT / 'harness/runs/q18a_alpha_perf_declare.sh').read_text())
        new = queue.derive()[DECLARE]
        diff = [line for line in difflib.ndiff(old.splitlines(), new.splitlines()) if line[:1] in '+-']
        self.assertEqual(len([x for x in diff if x[0] == '-']), 1)
        self.assertIn('+ ' + queue.DECLARE_FUSED.strip(), diff)
        self.assertIn('e3_step_a_box.py step_a_gate.py', new)

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
        # Every removed line names alpha-perf's gate or grade; each has its step-a line.
        self.assertEqual(len(removed), 7, removed)
        for line in removed:
            self.assertTrue('alpha_perf_gate' in line or 'alpha-perf-grade' in line, line)
            self.assertTrue(any('step_a_gate' in x or 'step-a-grade' in x for x in added), line)
        declared = queue.HEADER_NOTE.splitlines() + [queue.DRAFT_FUSED.strip(), '  e3_step_a_box.py step_a_gate.py; do',
                                                     queue.INTERFACE_NEW.splitlines()[0]]
        self.assertEqual([x for x in added if not ('step_a_gate' in x or 'step-a-grade' in x) and x not in declared], [], diff)

    def test_the_draft_reads_step_a_only(self):
        draft = queue.derive()[DRAFT]
        for text in ('from step_a_gate import window_ok', 'from step_a_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,PHASES',
                     "perf=p.parent/'step-a-grade.json'", 'python3 "$H/e3_step_a_box.py" arm', 'python3 "$H/e3_step_a_box.py" grade',
                     "assert env.get('FLASH_S5_QSA_FUSED') == '1'"):
            self.assertIn(text, draft)
        for text in ('from alpha_perf_gate import', 'python3 "$H/e3_alpha_perf_box.py"', 'q18a_alpha_perf', 'alpha-perf-grade'):
            self.assertNotIn(text, draft)
        self.assertEqual(queue.BODY.count('engine_watched '), 3)
        self.assertEqual(len(re.findall(r'^\s*python3 "\$H/', queue.BODY, flags=re.M)), 0)
        self.assertIn(tap.queue.GUARD_NEW, draft.replace('step_a', 'alpha_perf').replace('q18a-step-a', 'q18a-alpha-perf')
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
        self.assertEqual((gate.CLEANUP_S, gate.NOT_BEFORE, gate.HARD_END), (n40.CLEANUP_S, n40.NOT_BEFORE, n40.HARD_END))
        self.assertEqual(gate.HARD_END, 1793599200.0)      # 2026-11-02T06:00Z, inherited, not moved
        for name in n41.KEPT:
            self.assertEqual(gate.PHASES[name], n40.PHASES[name], name)
        self.assertEqual(gate.ACTIVE_S + gate.CLEANUP_S, gate.BOX_BUDGET_S)
        self.assertEqual(gate.BOX_BUDGET_S, 9000)
        self.assertEqual((gate.ARMS, len(queue.ARMS), gate.BAND1_ARMS, gate.BAND2_ARMS), (12, 12, 8, 4))
        self.assertEqual(gate.ARM_RESERVE_S, 450 + 1500 + 30 + 60)
        self.assertEqual(gate.LAST_ARM_ENTRY_S + gate.ARM_RESERVE_S + gate.TAIL_RESERVE_S, gate.ACTIVE_S)
        self.assertTrue(gate.arm_fits(gate.LAST_ARM_ENTRY_S))
        self.assertFalse(gate.arm_fits(gate.LAST_ARM_ENTRY_S + 1))
        self.assertLess(gate.EXPECTED_RUNTIME_S['nominal'], gate.BOX_BUDGET_S)
        body = queue.BODY
        for text, count in (('timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments', 2),
                            ('timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle', 2),
                            ('timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_step_a_box.py" arm', 1),
                            ('timeout 60s python3 "$H/e3_native39_trunk_checks.py" ingress', 1),
                            ('timeout 600s python3 "$H/e3_step_a_box.py" grade', 1)):
            self.assertEqual(body.count(text), count, text)
        self.assertEqual((n41.INSTRUMENTS_CAP_S, n41.FULL_IDLE_CAP_S, n41.INGRESS_CAP_S, gate.GRADE_CAP_S), (120, 450, 60, 600))
        self.assertEqual(gate.latest_arm(), gate.HARD_END - gate.BOX_BUDGET_S)
        self.assertEqual(time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime(gate.latest_arm())), '2026-11-02T03:30Z')
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
        self.assertIn('step_a_gate', slot)
        done = subprocess.run(['bash', '-c', 'fail_early() { return 2; }\n' + slot],
                              env=dict(os.environ, HARNESS_DIR=str(HERE)), capture_output=True, text=True)
        expected = 0 if gate.window_ok(time.time()) else 2
        self.assertEqual(done.returncode, expected, done.stderr)
        SLOT_RECEIPTS.append(dict(draft='q18a_step_a.draft.sh', real_wall_clock=True,
                                  verdict='PASS' if done.returncode == 0 else 'OUTSIDE-WINDOW',
                                  active_s=gate.ACTIVE_S, cleanup_margin_s=gate.CLEANUP_S))


if __name__ == '__main__':
    unittest.main()
