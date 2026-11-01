"""Queue-only CPU seams; native40's separate real Core probe checks the engine."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import copy
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'tools'), str(HERE.parent / 'patches/s5_engine')]
import e3_native40_unarmed_box as box
import e3_native39_w1 as w1
import native40_box_gate as gate
import native40_box_queue as queue
import e3_native40_regression as regression
SLOT_RECEIPTS = []


class Client:
    def __init__(self, mode=''):
        self.mode, self.actions, self.poll = mode, [], 0
        self.full = threading.Event()

    def enter(self, phase):
        self.actions.append(('enter', phase))

    def resume(self):
        self.actions.append(('resume',))
        return {'resumed': True}

    def call(self, action):
        self.actions.append((action,))
        self.poll += 1
        returned = self.poll >= 4
        row = dict(route='full' if returned else 'split', route_generation=3 if returned else 2,
                   admission_intent=None, scheduler_requests=0 if returned else 1,
                   transition_pending=False, observed_clock_ns=time.monotonic_ns(), clock=w1.clock_identity())
        receipt = None
        if returned:
            receipt = dict(reason='unarmed-stock-prefill-hold-timeout', admission_armed=False,
                           engine_epoch=7, route_generation=2, bound_ns=30_000_000_000, held_ns=30_000_000_000)
            if self.mode == 'armed':
                receipt['admission_armed'] = True
            if self.mode == 'early':
                receipt['held_ns'] = 1
            if self.mode == 'wrong-generation':
                receipt['route_generation'] = 1
            self.full.set()
        if self.mode == 'no-hold':
            count = 0
        else:
            count = self.poll - 1
        row['native_admission'] = dict(split_stock_deferrals=count, last_split_hold_timeout=receipt)
        return row

    def pause(self):
        return {}

    def leave(self):
        self.full.set()
        return {'route': 'full'}

    def send(self, url, model, result, timeout):
        if self.mode != 'client-error':
            self.full.wait(2)
            result.update(http_status=200, choices=1, completion_tokens=1, completed_ns=time.monotonic_ns())
        else:
            result['error_type'] = 'TimeoutError'


class Queue(unittest.TestCase):
    def exercise(self, mode=''):
        client = Client(mode)
        args = argparse.Namespace(url='http://127.0.0.1:8000', model='fixture', run_id='fixture', epoch=7,
                                  control_timeout_s=30, timeout_s=100)
        result = box.run(args, client=client, send=client.send, sleep=lambda _: None,
                         idle=lambda _: {'verdict': 'PASS'})
        self.assertEqual(client.actions[0], ('enter', 'split-w1-decode'))
        self.assertFalse(any('prefill_begin' in action for action in client.actions))
        return result

    def test_plain_completion_only_after_unarmed_bound_and_full(self):
        result = self.exercise()
        self.assertEqual(result['verdict'], 'PASS', result)
        self.assertTrue(result['no_native_arm'])
        self.assertEqual(result['automatic_return']['route'], 'full')

    def test_no_hold_cannot_pass(self):
        self.assertEqual(self.exercise('no-hold')['cause'], 'automatic-full-return-unproved')

    def test_armed_or_early_or_stale_return_cannot_pass(self):
        for mode in ('armed', 'early', 'wrong-generation'):
            with self.subTest(mode=mode):
                self.assertEqual(self.exercise(mode)['cause'], 'automatic-full-return-unproved')

    def test_client_failure_cannot_pass(self):
        self.assertNotEqual(self.exercise('client-error')['verdict'], 'PASS')

    def test_exact_queue_cli_through_parser_and_receipt(self):
        draft = queue.derive()['harness/runs/q18a_native40_unarmed_hold.draft.sh']
        line = draft[draft.index('  --url "$PROBE_URL" --model "$MODEL" --run-id "$RUN_ID" --epoch "$EPOCH"',
                                 draft.index('e3_native40_unarmed_box.py" \\')):]
        self.assertIn('--timeout-s 100 --control-timeout-s 30', line.split('rc=$?', 1)[0])
        client = Client()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'result.json'
            with mock.patch.object(box, 'NativeRuntimeClient', return_value=client), \
                    mock.patch.object(box, 'plain_request', client.send), \
                    mock.patch.object(box, 'full_idle', return_value={'verdict': 'PASS'}), contextlib.redirect_stdout(io.StringIO()):
                rc = box.main(['--url', 'http://127.0.0.1:8000', '--model', 'fixture', '--run-id', 'fixture',
                               '--epoch', '7', '--timeout-s', '100', '--control-timeout-s', '30', '--out', str(out)])
            self.assertEqual(rc, 0, json.loads(out.read_text()))
            self.assertEqual(json.loads(out.read_text())['verdict'], 'PASS')

    def test_exact_derivation_shell_and_executable_window(self):
        for name, text in queue.derive().items():
            path = HERE.parent / name
            self.assertEqual(path.read_text(), text)
            self.assertEqual(subprocess.run(['bash', '-n', str(path)], capture_output=True).returncode, 0)
        self.assertTrue(gate.window_ok(time.time()))
        self.assertFalse(gate.window_ok(gate.NOT_BEFORE - 1))
        self.assertFalse(gate.window_ok(gate.HARD_END - gate.ACTIVE_S - gate.CLEANUP_S + 1))
        draft = queue.derive()['harness/runs/q18a_native40_unarmed_hold.draft.sh']
        self.assertLess(draft.index('unarmed-stock-prefill "$rc"'), draft.index('qualification_phase row1'))
        self.assertIn('"$(native40_budget active)s"', draft)
        self.assertEqual(gate.ACTIVE_S, sum(gate.PHASES.values()))
        self.assertNotIn('qualification_phase rule4_controls\n', draft)
        self.assertEqual(draft.count('--row1 "$Q/qual-row1.json"'), 1)
        start = draft.index("python3 - <<'PY_SLOT'")
        slot = draft[start:draft.index('\nPY_SLOT\n', start) + len('\nPY_SLOT\n')]
        done = subprocess.run(['bash', '-c', 'fail_early() { return 2; }\n' + slot],
                              env=dict(os.environ, HARNESS_DIR=str(HERE)), capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        SLOT_RECEIPTS.append(dict(draft='q18a_native40_unarmed_hold.draft.sh', real_wall_clock=True,
                                  verdict='PASS', active_s=gate.ACTIVE_S, cleanup_margin_s=gate.CLEANUP_S))

    def test_regression_regrades_real_driver_source_and_rejects_cached_complete(self):
        from test_e3_native_measurement_clients import NativeWindowSource
        NativeWindowSource.setUpClass()
        row = copy.deepcopy(NativeWindowSource.record)
        row.update(complete=True, request={'prompt_tokens_observed': 2050, 'ttft_s': 1},
                   decode_window={'streams': [{'completion_delta': 1} for _ in range(8)]})
        data = dict(row=row, engine_epoch=NativeWindowSource.epoch, cleanup={'full_restored': True})
        self.assertEqual(regression.grade(data, 2050)['verdict'], 'PASS')
        row['end']['raw_journal_evidence'] = {}
        self.assertNotEqual(regression.grade(data, 2050)['verdict'], 'PASS')

    def test_regression_final_grade_requires_instruments_ingress_and_no_speed_claim(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ('unarmed-stock-prefill', 'qual-row1', 'instruments-begin', 'instruments-end', 'ingress-fresh', 'engine-alive'):
                (root / (name + '.json')).write_text(json.dumps({'verdict': 'PASS'}))
            path = root / 'w2-fresh.json'
            path.write_text(json.dumps({'grade': {'verdict': 'PASS', 'speed_claim': False}}))
            self.assertEqual(regression.final_grade(root)['verdict'], 'PASS')
            (root / 'ingress-fresh.json').write_text('{"verdict":"FAIL"}')
            self.assertEqual(regression.final_grade(root)['verdict'], 'INCONCLUSIVE')

    def test_regression_real_cli_through_package_and_native_client_refusal(self):
        import test_native35_performance as fixture
        server = fixture.LoopbackAPIServer()
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'row1.json').write_text('{"verdict":"PASS"}')
            env = fixture.round_env(fixture.w2_argv(root, server.url, 0)[0])
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(regression.perf.streams, '_start_streams', return_value=[]), \
                    mock.patch.object(regression.perf.streams, '_wait_open', return_value={'complete': True}):
                rc = regression.main(['--url', server.url, '--model', 'fixture', '--run-id', fixture.RUN_ID,
                    '--epoch', '7', '--workload', str(HERE / 'workloads/q18a-w2-overlap-native39.json'),
                    '--row1', str(root / 'row1.json'), '--out', str(root / 'regression.json')])
            data = json.loads((root / 'regression.json').read_text())
            self.assertEqual(rc, 2)
            self.assertIn('prefill_begin', [r['action'] for r in server.commands])
            self.assertIn('leave_full', [r['action'] for r in server.commands])
            self.assertEqual(data['grade']['verdict'], 'INCONCLUSIVE')


if __name__ == '__main__':
    unittest.main()
