"""Native35 L2: row4 client window at the accepted resume receipt.

Proves, on CPU: (1) the L2 runner's ``run`` is the live parent ``run`` with one
block moved; (2) with the real ``_wait_open``/``_wait_window``/``_prefill_progress``
and the box-shaped timing (decode cohort finishes before the native prefill's first
token) the parent refuses ``placement-window-incomplete`` with all eight streams
early-finished while L2 completes the window and the run; the fast-prefill case
passes both; L2 still refuses an early-finished stream and a prefill that never
opens; (3) the exact L2 queue row4 argv reaches its first network call through the
real CLI with the L2 runner; (4) the L2 queue is the L queue plus that one change;
(5) the L2 self-run probe and wrapper are the native35 ones plus the declared swaps.
"""
import datetime
import hashlib
import http.client
import inspect
import json
import os
from pathlib import Path
import shlex
from string import Template
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

import e3_decode_correctness as correctness
import e3_native_qualification_driver as driver_module
from e3_native_qualification_driver import (
    ImmutableStore, NativePlacementRunner, QualificationRefused,
    PLACEMENT_GENERATION_BUDGET, canonical_sha)
import e3_native35_l2_placement as l2
import test_e3_native_qualification_driver as base

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
SLOT_RECEIPTS = []
CLI_RECEIPTS = []
TIMING_RECEIPTS = []

OPEN_BLOCK = ("            prefill_api['open'] = correctness._wait_open(\n"
              "                prefill_states, prefill['workload'])\n"
              "            if prefill_api['open'].get('complete') is not True:\n"
              "                raise QualificationRefused('native-prefill-not-opened')\n")
WINDOW_BLOCK = ("            starts = {state.index: state.count() for state in states}\n"
                "            window = correctness._wait_window(states, starts, self.workload)\n"
                "            record['window'] = window\n"
                "            if window.get('complete') is not True:\n"
                "                raise QualificationRefused('placement-window-incomplete')\n")
RESUME_ACCEPTED = ("                raise QualificationRefused('resume-not-accepted')\n")


def real_prefill_plan(open_s=None):
    plan = json.loads((HERE / 'workloads/q18a-native-placement-prefill.json').read_text())
    if open_s is not None:
        plan['workload']['timeouts']['open_s'] = open_s
    return plan


class SourceRelation(unittest.TestCase):
    def test_run_is_the_parent_run_with_only_the_window_block_moved(self):
        parent = inspect.getsource(NativePlacementRunner.run)
        child = inspect.getsource(l2.ResumeWindowPlacementRunner.run)
        self.assertEqual(parent.count(OPEN_BLOCK + WINDOW_BLOCK), 1)
        self.assertEqual(parent.count(RESUME_ACCEPTED + OPEN_BLOCK), 1)
        self.assertEqual(child, parent.replace(OPEN_BLOCK + WINDOW_BLOCK, WINDOW_BLOCK + OPEN_BLOCK))
        # The window now follows the accepted resume directly.
        self.assertEqual(child.count(RESUME_ACCEPTED + WINDOW_BLOCK), 1)
        own = set(vars(l2.ResumeWindowPlacementRunner)) - {'__module__', '__doc__', '__qualname__',
                                                           '__firstlineno__', '__static_attributes__'}
        self.assertEqual(own, {'run'})
        self.assertTrue(issubclass(l2.ResumeWindowPlacementRunner, NativePlacementRunner))
        # Every module name the copied body resolves is the driver module's object.
        names = set(l2.ResumeWindowPlacementRunner.run.__code__.co_names)
        for name in sorted(n for n in names & set(vars(driver_module)) if not n.startswith('__')):
            if name in vars(l2):
                self.assertIs(vars(l2)[name], vars(driver_module)[name], name)
        self.assertIs(l2.correctness, correctness)
        self.assertEqual(correctness.WINDOW_BASIS.split(';')[0], 'start is the accepted resume receipt')


class Timeline:
    """Engine progress after the in-window resume: lockstep decode tokens and one
    native prefill whose first token arrives at a declared point."""

    def __init__(self, decode, prefill, *, tick_s=0.005, open_tick=None,
                 open_after_decode_s=None):
        self.decode, self.prefill = decode, prefill
        self.tick_s, self.open_tick, self.open_after_decode_s = tick_s, open_tick, open_after_decode_s
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.decode_done_ns = None
        self.prefill_open_ns = None

    def start(self):
        self.thread.start()

    def _run(self):
        tick = 0
        decode_done = None
        while not self.stop.is_set():
            time.sleep(self.tick_s)
            tick += 1
            now = time.monotonic_ns()
            for state in self.decode:
                if state.finish_at_ns is None:
                    with state.lock:
                        state.value += 1
                        if state.value >= PLACEMENT_GENERATION_BUDGET:
                            state.finish_at_ns = now
                            state.finish_reason = 'length'
                    state.progress.set()
            if decode_done is None and all(s.finish_at_ns is not None for s in self.decode):
                decode_done = time.monotonic()
                self.decode_done_ns = now
            opened = self.prefill_open_ns is not None
            due = ((self.open_tick is not None and tick >= self.open_tick)
                   or (self.open_after_decode_s is not None and decode_done is not None
                       and time.monotonic() - decode_done >= self.open_after_decode_s))
            if not opened and due:
                for state in self.prefill:
                    with state.lock:
                        state.first_advance_ns = time.monotonic_ns()
                    state.progress.set()
                self.prefill_open_ns = time.monotonic_ns()
            never = self.open_tick is None and self.open_after_decode_s is None
            if decode_done is not None and (self.prefill_open_ns is not None or never):
                return


class TimelineDriver(base.PlacementDriver):
    """The test_e3_native_qualification_driver control fake; the second resume
    (after the in-window prefill submission) starts the declared timeline."""

    def __init__(self, timeline, **kwargs):
        super().__init__(**kwargs)
        self.timeline = timeline
        self.resumes = 0

    def _session(self, phase_name, action, **fields):
        reply = super()._session(phase_name, action, **fields)
        if action == 'resume':
            self.resumes += 1
            if self.resumes == 2:
                self.timeline.start()
        return reply


def run_case(runner_cls, *, initial, open_tick=None, open_after_decode_s=None, plan=None):
    states = [base.FakeState(index) for index in range(8)]
    for state, value in zip(states, initial):
        state.value = value
    prefill = [base.FakeState(0, prompt_tokens=4167)]
    prefill[0].first_advance_ns = None
    timeline = Timeline(states, prefill, open_tick=open_tick, open_after_decode_s=open_after_decode_s)
    driver = TimelineDriver(timeline)
    runner = runner_cls(workload=base.placement_workload(), prefill_plan=plan or real_prefill_plan(),
                        url='http://127.0.0.1:9')
    windows = []
    real_wait_window = correctness._wait_window

    def recording_wait_window(states_, starts, workload):
        window = real_wait_window(states_, starts, workload)
        windows.append(window)
        return window

    def start_streams(url, model, bodies, cohort, workload, **kwargs):
        return prefill if len(bodies) == 1 else states

    def stop_streams(record, states_, workload):
        record['readers_alive_after_join'] = []
        return []

    outcome = None
    try:
        with mock.patch.object(correctness, '_start_streams', start_streams), \
             mock.patch.object(correctness, '_stop_streams', stop_streams), \
             mock.patch.object(correctness, '_wait_window', recording_wait_window):
            try:
                outcome = runner.run(driver, window_id='window-1', run_id='run-1',
                                     engine_epoch=3, model='m')
            except QualificationRefused as exc:
                outcome = exc
    finally:
        timeline.stop.set()
        if timeline.thread.is_alive() or timeline.thread.ident is not None:
            timeline.thread.join(5)
    return outcome, driver, windows, timeline


class Timing(unittest.TestCase):
    """Real _wait_open/_wait_window/_prefill_progress and the real frozen workload
    and prefill plan; only the HTTP streams and the engine control are fixtures."""

    BOX_INITIAL = (1, 2, 3, 4, 5, 6, 7, 8)

    def test_box_shaped_timing_parent_refuses_and_l2_completes(self):
        parent, parent_driver, _, _ = run_case(NativePlacementRunner, initial=self.BOX_INITIAL,
                                               open_after_decode_s=0.1)
        self.assertIsInstance(parent, QualificationRefused)
        self.assertEqual(parent.code, 'placement-window-incomplete')
        window = parent.evidence['client_window']
        self.assertEqual((window['complete'], window['reason']), (False, 'early_finish'))
        self.assertEqual(window['early_finish'], list(range(8)))
        self.assertEqual({(r['start_completion'], r['end_completion']) for r in window['streams']},
                         {(PLACEMENT_GENERATION_BUDGET, PLACEMENT_GENERATION_BUDGET)})
        self.assertGreater(window['start_ns'], parent.evidence['prefill_request']['open']['sealed_at_ns'])
        self.assertTrue(parent.cleanup['restored'])

        child, driver, windows, timeline = run_case(l2.ResumeWindowPlacementRunner,
                                                    initial=self.BOX_INITIAL, open_after_decode_s=0.1)
        self.assertIsInstance(child, dict, getattr(child, 'code', child))
        self.assertEqual(len(windows), 1)
        window = windows[0]
        self.assertEqual((window['complete'], window['reason'], window['early_finish']), (True, 'token_bound', []))
        self.assertTrue(all(r['completion_delta'] >= 40 for r in window['streams']))
        self.assertTrue(all(r['start_completion'] <= PLACEMENT_GENERATION_BUDGET - 40 for r in window['streams']))
        opened = child['native_prefill_request']['open']
        self.assertTrue(opened['complete'])
        # Window first, then the prefill's first token, which arrived after the cohort ended.
        self.assertLess(window['end_ns'], opened['sealed_at_ns'])
        self.assertLess(timeline.decode_done_ns, timeline.prefill_open_ns)
        self.assertTrue(child['native_prefill_request']['complete'])
        self.assertTrue(child['native_prefill_request']['token_count_proved'])
        # The engine-facing protocol is the parent's, call for call.
        self.assertEqual(driver.calls, ['prepare', 'resume', 'pause_enter_split', 'overlap_begin',
                                        'resume', 'overlap_end', 'leave_full', 'resume'])
        self.assertEqual(driver.worker_calls, ['teacher_configure', 'observer_begin', 'observer_end',
                                               'teacher_evidence', 'teacher_configure'])
        self.assertEqual(parent_driver.calls[:5], driver.calls[:5])
        TIMING_RECEIPTS.append(dict(case='box-shaped', parent='placement-window-incomplete',
                                    parent_early_finish=list(range(8)), l2='COMPLETE',
                                    l2_window_reason=window['reason'], real_wait_functions=True))

    def test_fast_prefill_passes_both_runners_with_identical_calls(self):
        calls = {}
        for cls in (NativePlacementRunner, l2.ResumeWindowPlacementRunner):
            outcome, driver, windows, _ = run_case(cls, initial=self.BOX_INITIAL, open_tick=1)
            self.assertIsInstance(outcome, dict, (cls.__name__, getattr(outcome, 'code', outcome)))
            self.assertEqual(len(windows), 1)
            self.assertTrue(windows[0]['complete'], cls.__name__)
            calls[cls.__name__] = (driver.calls, driver.worker_calls)
        self.assertEqual(calls['NativePlacementRunner'], calls['ResumeWindowPlacementRunner'])
        TIMING_RECEIPTS.append(dict(case='fast-prefill', parent='COMPLETE', l2='COMPLETE'))

    def test_l2_still_refuses_an_early_finished_stream(self):
        # Stream 0 already holds 30 tokens: it ends before the others gain 40.
        outcome, _, windows, _ = run_case(l2.ResumeWindowPlacementRunner,
                                          initial=(30, 1, 2, 3, 4, 5, 6, 7), open_tick=1)
        self.assertIsInstance(outcome, QualificationRefused)
        self.assertEqual(outcome.code, 'placement-window-incomplete')
        window = outcome.evidence['client_window']
        self.assertEqual((window['complete'], window['reason'], window['early_finish']),
                         (False, 'early_finish', [0]))
        self.assertTrue(outcome.cleanup['restored'])
        TIMING_RECEIPTS.append(dict(case='early-finish', l2='placement-window-incomplete'))

    def test_l2_still_refuses_a_prefill_that_never_opens(self):
        for cls, window_seen in ((NativePlacementRunner, False), (l2.ResumeWindowPlacementRunner, True)):
            outcome, _, windows, _ = run_case(cls, initial=self.BOX_INITIAL,
                                              plan=real_prefill_plan(open_s=0.5))
            self.assertIsInstance(outcome, QualificationRefused, cls.__name__)
            self.assertEqual(outcome.code, 'native-prefill-not-opened', cls.__name__)
            self.assertEqual(bool(windows), window_seen, cls.__name__)
            if window_seen:
                self.assertTrue(outcome.evidence['client_window']['complete'])
            self.assertTrue(outcome.cleanup['restored'])
        TIMING_RECEIPTS.append(dict(case='prefill-never-opens', parent='native-prefill-not-opened',
                                    l2='native-prefill-not-opened after a complete window'))


L_HEADER = ('# Native35-l: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 -> native lifetime -> hygiene/leak -> owned restore.\n')
L2_HEADER = ('# Native35-l2: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 (client window at the accepted resume receipt) -> native lifetime -> hygiene/leak -> owned restore.\n')
L_PHASE = ('  local name=$1 rc\n'
           '  python3 "$H/e3_q18a_qualification.py" --phase "$name" --url "$PROBE_URL" \\\n')
L2_PHASE = ('  local name=$1 rc cli=e3_q18a_qualification.py\n'
            '  # L2: row4 uses the resume-receipt window collector; argv and record are unchanged.\n'
            '  if [[ $name == row4 ]]; then cli=e3_native35_l2_placement.py; fi\n'
            '  python3 "$H/$cli" --phase "$name" --url "$PROBE_URL" \\\n')


def l2_names(text):
    return text.replace('q18a-native35-l', 'q18a-native35-l2').replace('q18a_native35_l', 'q18a_native35_l2')


class Queue(unittest.TestCase):
    def test_l2_is_the_l_queue_plus_the_row4_collector_branch(self):
        draft = (RUNS / 'q18a_native35_l.draft.sh').read_text()
        self.assertEqual(draft.count(L_HEADER), 1)
        self.assertEqual(draft.count(L_PHASE), 1)
        expected = l2_names(draft.replace(L_HEADER, L2_HEADER).replace(L_PHASE, L2_PHASE))
        actual = (RUNS / 'q18a_native35_l2.draft.sh').read_text()
        self.assertEqual(actual, expected)
        self.assertEqual(actual.count('e3_native35_l2_placement.py'), 1)
        self.assertEqual(actual.count('python3 "$H/$cli"'), 1)
        self.assertNotIn('e3_native34_causal.py', actual)
        for tool in ('e3_q18a_lifecycle.py', 'leaktest.py" 8 3', 'pressure2.py', 'leak_partial_tail.py',
                     'pp_release_grade.py" leak', 'pp_release_grade.py" lifetime', 'pressure_wait.py',
                     'native35_lifetime_grade.py', 'native35_ledger_grade.py', '7200s env'):
            self.assertIn(tool, actual)
        for suffix in ('_settings.sh', '_declare.sh'):
            self.assertEqual((RUNS / ('q18a_native35_l2' + suffix)).read_text(),
                             l2_names((RUNS / ('q18a_native35_l' + suffix)).read_text()))
        settings = (RUNS / 'q18a_native35_l2_settings.sh').read_text()
        self.assertIn(' FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=0 ', settings)
        self.assertNotIn('FLASH_NATIVE34_ARM', settings)
        self.assertNotIn('READBACK', settings)
        self.assertIn('QNAME=${QNAME:-q18a-native35-l2}', settings)

    def test_every_l2_script_parses_and_no_global_blocking(self):
        for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
            path = RUNS / ('q18a_native35_l2' + suffix)
            self.assertEqual(subprocess.run(['bash', '-n', str(path)], capture_output=True).returncode, 0, path.name)
            self.assertNotIn('CUDA_LAUNCH_BLOCKING=1', path.read_text())

    def test_l2_draft_window_slot_executes_at_real_wall_clock(self):
        path = RUNS / 'q18a_native35_l2.draft.sh'
        source = path.read_text()
        begin = source.index("python3 - <<'PY_SLOT'")
        end = source.index('\nPY_SLOT', begin) + len('\nPY_SLOT')
        actual = source[begin:end]
        self.assertIn('from native35_gate import lifetime_window_ok as window_ok', actual)
        self.assertNotIn('datetime', actual)
        self.assertIn('exec timeout --signal=TERM --kill-after=420s 7200s env', source)
        env = dict(os.environ, HARNESS_DIR=str(HERE), PYTHONDONTWRITEBYTECODE='1')
        completed = subprocess.run(['bash', '-c', 'fail_early() { return 2; }\n' + actual],
                                   env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        SLOT_RECEIPTS.append(dict(draft=path.name, verdict='PASS', rc=0,
                                  slot_sha256=hashlib.sha256(actual.encode()).hexdigest(),
                                  real_wall_clock=True,
                                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  engine_contact=False, active_s=7200, cleanup_margin_s=420))


def queue_row4_argv(directory):
    text = (RUNS / 'q18a_native35_l2.draft.sh').read_text()
    invocation = text.split('python3 "$H/$cli"', 1)[1].split('> "$Q/qual-$name.log"', 1)[0]
    values = dict(H=str(HERE), Q=str(directory), PROBE_URL='http://127.0.0.1:1', MODEL='qwen38-w8a8-tp4',
                  RUN_ID='q18a-native35-l2-cpu', EPOCH='7', Q18A_STORE=str(directory / 'private/qualification-store'),
                  SNAPSHOT_MAX_BYTES=str(512 * 1024 ** 2), name='row4')
    return [Template(token).substitute(values) for token in shlex.split(invocation.replace('\\\n', ' '))], values


class QueueCLI(unittest.TestCase):
    def test_exact_row4_argv_reaches_first_network_call_with_the_l2_runner(self):
        fixture_key = 'native35-l2-cpu-transport-fixture'
        seen = []

        def refuse_request(connection, method, url, body=None, headers=None, **kwargs):
            command = json.loads(body)
            runners = [type(frame.frame.f_locals['self']) for frame in inspect.stack()
                       if frame.function == 'run' and isinstance(frame.frame.f_locals.get('self'), NativePlacementRunner)]
            seen.append(dict(host=connection.host, port=connection.port, method=method, url=url,
                             action=command.get('action'), target_route=command.get('target_route'),
                             epoch=command.get('engine_epoch'), runners=runners,
                             authorized=headers.get('Authorization') == 'Bearer ' + fixture_key))
            raise ConnectionRefusedError('declared CPU boundary')

        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            argv, values = queue_row4_argv(directory)
            self.assertEqual(argv[:2], ['--phase', 'row4'])
            # Declared leaves: row1 and rule4_controls run before row4 on the box and leave the
            # frozen row1 plan (parsed, unused by row4) and the retained teacher artifact.
            (directory / 'private').mkdir()
            (directory / 'private/row1-plan.json').write_text(json.dumps({'schema': 'cpu-fixture-row1-plan'}))
            artifact = base.retained_teacher(ImmutableStore(values['Q18A_STORE']), {'schema': 'cpu-fixture', 'rows': []})
            env = dict(VLLM_API_KEY=fixture_key, Q18A_TEACHER_ARTIFACT_SHA=canonical_sha(artifact),
                       Q18A_ROW1_SHA='e' * 64)
            with mock.patch.dict(os.environ, env), mock.patch('http.client.HTTPConnection.request', refuse_request):
                rc = l2.main(argv)
            self.assertIs(driver_module.NativePlacementRunner, NativePlacementRunner)
            record = json.loads((directory / 'qual-row4.json').read_text())
        self.assertEqual(rc, 2)
        self.assertEqual(record['verdict'], 'INCONCLUSIVE')
        self.assertEqual(len(seen), 1, seen)
        first = seen[0]
        self.assertEqual((first['host'], first['port'], first['method'], first['url']),
                         ('127.0.0.1', 1, 'POST', '/v1/flash_s5_native_overlap'))
        self.assertEqual((first['action'], first['target_route'], first['epoch']), ('prepare', 'full', 7))
        self.assertTrue(first['authorized'])
        self.assertEqual(first['runners'], [l2.ResumeWindowPlacementRunner])
        self.assertEqual(len(record['missing']), 1)
        self.assertTrue(record['missing'][0].startswith('collect:'), record['missing'])
        self.assertNotIn(fixture_key, json.dumps(record))
        paths = [HERE / 'workloads/q17b-correctness-precontrols.json', HERE / 'workloads/q18a-native-placement-prefill.json']
        CLI_RECEIPTS.append(dict(case='exact-l2-row4-argv', verdict='PASS', first_network_call='prepare-full',
                                 network_calls=len(seen), network_contact=False, runner='ResumeWindowPlacementRunner',
                                 expected_refusal=record['missing'][0], substitution_restored=True,
                                 workload_and_plan_shas={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))

    def test_other_phases_and_the_parent_cli_are_untouched(self):
        import e3_q18a_qualification as qualification
        self.assertIs(driver_module.NativePlacementRunner, NativePlacementRunner)
        self.assertNotIn('e3_native35_l2_placement', inspect.getsource(qualification))
        text = (RUNS / 'q18a_native35_l2.draft.sh').read_text()
        phase = text[text.index('qualification_phase() {'):text.index('\n}\n', text.index('qualification_phase() {'))]
        self.assertIn('if [[ $name == row4 ]]; then cli=e3_native35_l2_placement.py; fi', phase)
        self.assertIn('local name=$1 rc cli=e3_q18a_qualification.py', phase)


class Tools(unittest.TestCase):
    def test_l2_selfrun_tools_are_the_native35_tools_plus_the_declared_swaps(self):
        # Hub: <repo>/tools; pinned image: /source/tools (the bundle's native35_tools).
        import importlib.util
        tools = HERE.parent / 'tools'
        spec = importlib.util.spec_from_file_location('native35_l2_tools_derive', tools / 'native35_l2_tools_derive.py')
        derive = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(derive)
        generated = derive.generate(tools)
        self.assertEqual(sorted(p.name for p in generated), ['native35_l2_cpu_probe.py', 'native35_l2_selfrun.py'])
        for path, text in generated.items():
            self.assertEqual(path.read_text(), text, path.name)
        probe = (tools / 'native35_l2_cpu_probe.py').read_text()
        self.assertIn("pattern='test_native35_*.py'", probe)
        self.assertIn('EXPECTED_TESTS=136', probe)
        selfrun = (tools / 'native35_l2_selfrun.py').read_text()
        self.assertEqual(selfrun.count("'--network','none'"), 1)
        self.assertIn("'FLASH_NATIVE34_DIAG=1','FLASH_NATIVE34_ARM=O','FLASH_NATIVE34_READBACK=1'", selfrun)


if __name__ == '__main__':
    unittest.main()
