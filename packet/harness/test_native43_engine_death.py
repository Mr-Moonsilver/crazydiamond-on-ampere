"""Native43 le-fix (a)/(c): engine-death scan and watch over verbatim box log excerpts.

Fixtures: harness/fixtures/native43_engine_death (PROVENANCE.json names each source and line range).
Leaves: native43_engine_death_fixture_leaves.json.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import native43_engine_death as death  # noqa: E402

FIX = HERE / 'fixtures/native43_engine_death'
TOOL = HERE / 'native43_engine_death.py'
LE_SHUTDOWN = '2026-11-01T11:03:08.286892993Z'


def fixture_lines(name):
    return (FIX / name).read_bytes().split(b'\n')[:-1]


class Root(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='n43-death-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def write(self, name, lines):
        path = self.root / name
        path.write_bytes(b''.join(line + b'\n' for line in lines))
        return path

    def scan(self, logs, intent=None):
        out = self.root / ('scan-%d.json' % len(list(self.root.glob('scan-*.json'))))
        argv = [sys.executable, '-B', str(TOOL), 'scan', '--out', str(out)]
        for log in logs:
            argv += ['--log', str(log)]
        if intent is not None:
            argv += ['--stop-intent', str(intent)]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        return result.returncode, json.loads(out.read_text()), result.stdout


class Fixtures(unittest.TestCase):
    def test_excerpts_match_their_provenance(self):
        prov = json.loads((FIX / 'PROVENANCE.json').read_text())
        self.assertEqual(prov['schema'], 'native43-engine-death-fixtures/1')
        names = sorted(p.name for p in FIX.iterdir() if p.name != 'PROVENANCE.json')
        self.assertEqual(names, sorted(row['fixture'] for row in prov['fixtures']))
        for row in prov['fixtures']:
            self.assertEqual(hashlib.sha256((FIX / row['fixture']).read_bytes()).hexdigest(), row['fixture_sha256'])
            source = Path(prov['root']) / row['source']
            if source.is_file():  # the hub keeps the fetches; a copy elsewhere still checks the excerpt hashes
                data = source.read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), row['source_sha256'])
                if row['segments']:
                    lines = data.split(b'\n')
                    piece = b''.join(b'\n'.join(lines[a - 1:b]) + b'\n' for a, b in row['segments'])
                    self.assertEqual(piece, (FIX / row['fixture']).read_bytes())


class Scan(Root):
    def test_le_unplanned_teardown_is_dead_before_the_stop_intent(self):
        rc, rec, out = self.scan([FIX / 'le-engine-stream.log', FIX / 'le-engine-stream.err'],
                                 FIX / 'le-host-stop-intent.json')
        self.assertEqual((rc, rec['verdict']), (1, 'DEAD'))
        self.assertEqual(rec['first_death']['kind'], 'executor_teardown')
        self.assertEqual(rec['first_death']['wall'], LE_SHUTDOWN)
        self.assertIn('[shutdown] Executor: waiting for worker exit count=4', rec['first_death']['text'])
        self.assertEqual(rec['stop_intent']['state'], 'read')
        self.assertEqual(rec['first_orderly']['kind'], 'api_shutdown_triggered')
        # 3 executor lines + 1 parent-exited line + 42 broadcast stalls, all before the 11:30:52 stop.
        self.assertEqual(rec['death_counts'], {'executor_teardown': 3, 'worker_parent_exited': 1,
                                               'broadcast_block_stall': 42})
        self.assertIn('ENGINE_DEATH_SCAN DEAD first=executor_teardown', out)

    def test_le_at_capture_time_without_intent_is_dead(self):
        # capture_logs runs before stop_engine writes the intent: only the log decides.
        lines = fixture_lines('le-engine-stream.log')
        cut = next(i for i, line in enumerate(lines) if b'API server: shutdown triggered' in line)
        rc, rec, _ = self.scan([self.write('le-capture.log', lines[:cut])], self.root / 'absent.json')
        self.assertEqual((rc, rec['verdict'], rec['stop_intent']['state']), (1, 'DEAD', 'absent'))
        self.assertEqual(rec['first_death']['kind'], 'executor_teardown')

    def test_broadcast_stall_alone_is_dead(self):
        lines = [line for line in fixture_lines('le-engine-stream.log')
                 if b'[shutdown]' not in line and b'Parent process exited' not in line]
        rc, rec, _ = self.scan([self.write('stall.log', lines)])
        self.assertEqual((rc, rec['first_death']['kind']), (1, 'broadcast_block_stall'))
        self.assertEqual(rec['first_death']['wall'], '2026-11-01T11:04:06.066739045Z')

    def test_le_before_the_teardown_is_alive(self):
        lines = fixture_lines('le-engine-stream.log')
        cut = next(i for i, line in enumerate(lines) if b'[shutdown] Executor' in line)
        rc, rec, _ = self.scan([self.write('le-healthy.log', lines[:cut])])
        self.assertEqual((rc, rec['verdict'], rec['first_death']), (0, 'ALIVE', None))

    def test_trunk_orderly_stop_is_alive(self):
        rc, rec, _ = self.scan([FIX / 'trunk-engine-stream.log'], FIX / 'trunk-host-stop-intent.json')
        self.assertEqual((rc, rec['verdict']), (0, 'ALIVE'))
        self.assertEqual(rec['first_orderly']['kind'], 'api_shutdown_triggered')

    def test_trunk_orderly_lines_alone_bound_the_teardown(self):
        rc, rec, _ = self.scan([FIX / 'trunk-engine-stream.log'])
        self.assertEqual((rc, rec['verdict'], rec['stop_intent']['state']), (0, 'ALIVE', 'not-given'))

    def test_intent_alone_bounds_the_teardown(self):
        lines = [line for line in fixture_lines('trunk-engine-stream.log')
                 if b'API server: shutdown triggered' not in line and b'EngineCore: trigger received' not in line
                 and b'EngineCore: start mode' not in line]
        rc, rec, _ = self.scan([self.write('trunk-no-trigger.log', lines)], FIX / 'trunk-host-stop-intent.json')
        self.assertEqual((rc, rec['verdict'], rec['orderly_bound']), (0, 'ALIVE', '2026-11-01T03:07:28.663200742Z'))

    def test_trunk_teardown_without_any_trigger_is_dead(self):
        lines = [line for line in fixture_lines('trunk-engine-stream.log')
                 if b'API server: shutdown triggered' not in line and b'EngineCore: trigger received' not in line
                 and b'EngineCore: start mode' not in line]
        rc, rec, _ = self.scan([self.write('trunk-no-trigger.log', lines)])
        self.assertEqual((rc, rec['first_death']['kind']), (1, 'executor_teardown'))

    def test_invalid_intent_bounds_nothing(self):
        bad = self.root / 'intent.json'
        bad.write_text('{"wall_ns": "soon"}')
        lines = [line for line in fixture_lines('trunk-engine-stream.log')
                 if b'API server: shutdown triggered' not in line and b'EngineCore: trigger received' not in line
                 and b'EngineCore: start mode' not in line]
        rc, rec, _ = self.scan([self.write('t.log', lines)], bad)
        self.assertEqual((rc, rec['stop_intent']['state']), (1, 'invalid:ValueError'))

    def test_hard_markers_are_dead_anywhere(self):
        rc, rec, _ = self.scan([FIX / 'n33-engine-stream.log'], FIX / 'trunk-host-stop-intent.json')
        self.assertEqual((rc, rec['first_death']['kind']), (1, 'engine_dead'))
        self.assertEqual(set(rec['death_counts']), {'engine_dead', 'worker_died', 'executor_teardown',
                                                     'worker_parent_exited'})

    def test_pc_stall_precedes_the_fatal_error_by_four_minutes(self):
        rc, rec, _ = self.scan([FIX / 'pc-engine-stream.log'])
        self.assertEqual((rc, rec['first_death']['kind'], rec['first_death']['wall']),
                         (1, 'broadcast_block_stall', '2026-10-31T18:04:04.722139870Z'))
        self.assertEqual(rec['death_counts'], {'broadcast_block_stall': 1, 'engine_fatal': 1})

    def test_unreadable_logs_are_inconclusive(self):
        rc, rec, _ = self.scan([self.root / 'missing.log'])
        self.assertEqual((rc, rec['verdict']), (2, 'INCONCLUSIVE'))

    def test_untimestamped_teardown_needs_an_earlier_orderly_line(self):
        plain = [b'INFO [launcher.py:129] [shutdown] API server: shutdown triggered',
                 b'INFO [multiproc_executor.py:507] [shutdown] Executor: waiting for worker exit count=4']
        rc, rec, _ = self.scan([self.write('plain.log', plain)], FIX / 'trunk-host-stop-intent.json')
        self.assertEqual(rc, 0)
        rc, rec, _ = self.scan([self.write('plain2.log', plain[::-1])], FIX / 'trunk-host-stop-intent.json')
        self.assertEqual((rc, rec['first_death']['kind']), (1, 'executor_teardown'))

    def test_incremental_reads_split_mid_line_decide_like_one_read(self):
        data = (FIX / 'le-engine-stream.log').read_bytes()
        path = self.root / 'grow.log'
        scanner = death.Scanner([path])
        path.write_bytes(b'')
        step = 997
        for start in range(0, len(data), step):
            with path.open('ab') as handle:
                handle.write(data[start:start + step])
            scanner.poll()
        scanner.poll(final=True)
        one = death.Scanner([FIX / 'le-engine-stream.log'])
        one.poll(final=True)
        a, b = scanner.decide(), one.decide()
        self.assertEqual((a['verdict'], a['first_death']['text'], a['death_counts']),
                         (b['verdict'], b['first_death']['text'], b['death_counts']))
        self.assertEqual(a['files'][0]['sha256'], hashlib.sha256(data).hexdigest())


class Watch(Root):
    def fake(self, name, body):
        path = self.root / name
        path.write_text('#!/bin/sh\n' + body + '\n')
        path.chmod(0o755)
        return str(path)

    def healthy_gpus(self):
        return self.fake('gpu-ok', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')

    def run_watch(self, child, *, logs=(), extra=(), feeder=None, stdin=None, timeout=90):
        record = self.root / 'watch.json'
        argv = [sys.executable, '-B', str(TOOL), 'watch', '--record', str(record), '--poll-s', '0.2',
                '--kill-grace-s', '2']
        for log in logs or [self.root / 'engine.log']:
            argv += ['--log', str(log)]
        if '--gpu-cmd' not in extra:
            argv += ['--gpu-cmd', self.healthy_gpus()]
        argv += list(extra) + ['--'] + child
        started = time.monotonic()
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE if stdin else None, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        if feeder:
            threading.Thread(target=feeder, daemon=True).start()
        out, err = proc.communicate(stdin, timeout=timeout)
        rec = json.loads(record.read_text()) if record.exists() else None
        return proc.returncode, rec, out, err, time.monotonic() - started

    def test_le_teardown_during_a_hung_client_stops_it_within_seconds(self):
        log = self.root / 'engine.log'
        lines = fixture_lines('le-engine-stream.log')
        cut = next(i for i, line in enumerate(lines) if b'[shutdown] Executor' in line)
        log.write_bytes(b''.join(line + b'\n' for line in lines[:cut]))
        marker = self.root / 'client-started'

        def feed():
            deadline = time.monotonic() + 30
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(1.0)
            with log.open('ab') as handle:
                handle.write(lines[cut] + b'\n')
        child = [sys.executable, '-c', 'import pathlib,time,sys; pathlib.Path(sys.argv[1]).touch(); time.sleep(600)',
                 str(marker)]
        rc, rec, _, err, wall = self.run_watch(child, logs=[log], feeder=feed)
        self.assertEqual((rc, rec['verdict'], rec['dead']['reason']), (1, 'DEAD', 'engine-log'))
        self.assertEqual(rec['dead']['first_death']['wall'], LE_SHUTDOWN)
        self.assertFalse(rec['stop']['killed'])
        self.assertEqual(rec['stop']['child_rc'], -signal.SIGTERM)
        self.assertLess(wall, 30)
        self.assertIn('ENGINE_DEATH_WATCH DEAD reason=engine-log', err)

    def test_healthy_engine_passes_the_client_code_through(self):
        log = self.write('engine.log', fixture_lines('trunk-engine-stream.log')[:6])
        rc, rec, _, _, _ = self.run_watch([sys.executable, '-c', 'import time; time.sleep(0.6); raise SystemExit(3)'],
                                          logs=[log])
        self.assertEqual((rc, rec['verdict'], rec['client_rc']), (3, 'CLIENT_EXIT', 3))
        self.assertGreaterEqual(rec['polls'], 1)

    def test_one_gpu_below_floor_for_two_polls_is_dead(self):
        gpu = self.fake('gpu-low', 'printf "0, 22925\\n1, 3\\n2, 23037\\n3, 22947\\n"')
        rc, rec, _, _, _ = self.run_watch([sys.executable, '-c', 'import time; time.sleep(600)'],
                                          extra=['--gpu-cmd', gpu])
        self.assertEqual((rc, rec['dead']['reason'], rec['dead']['consecutive_polls']), (1, 'gpu-below-floor', 2))
        self.assertEqual(rec['dead']['gpu_used_mib'], [22925, 3, 23037, 22947])

    def test_failing_gpu_query_is_not_death(self):
        gpu = self.fake('gpu-fail', 'exit 9')
        rc, rec, _, _, _ = self.run_watch([sys.executable, '-c', 'import time; time.sleep(1.0)'],
                                          extra=['--gpu-cmd', gpu])
        self.assertEqual((rc, rec['verdict']), (0, 'CLIENT_EXIT'))
        self.assertGreaterEqual(rec['gpu_errors'], 1)

    def test_stopped_or_absent_container_is_dead(self):
        for body, state in (('echo false', 'stopped'), ('echo "Error: No such object: x" >&2; exit 1', 'absent')):
            with self.subTest(state=state):
                (self.root / 'watch.json').unlink(missing_ok=True)
                docker = self.fake('docker-' + state, body)
                rc, rec, _, _, _ = self.run_watch([sys.executable, '-c', 'import time; time.sleep(600)'],
                                                  extra=['--container', 'qwen-flash-prod', '--docker', docker])
                self.assertEqual((rc, rec['dead']['reason'], rec['dead']['state']), (1, 'container-not-running', state))

    def test_running_container_and_docker_errors_are_not_death(self):
        for body in ('echo true', 'exit 7'):
            with self.subTest(body=body):
                (self.root / 'watch.json').unlink(missing_ok=True)
                docker = self.fake('docker-x', body)
                rc, rec, _, _, _ = self.run_watch([sys.executable, '-c', 'import time; time.sleep(1.0)'],
                                                  extra=['--container', 'qwen-flash-prod', '--docker', docker])
                self.assertEqual((rc, rec['verdict']), (0, 'CLIENT_EXIT'))

    def test_stall_rule_is_off_by_default_and_fires_when_set(self):
        log = self.write('engine.log', fixture_lines('trunk-engine-stream.log')[:6])
        rc, rec, _, _, _ = self.run_watch([sys.executable, '-c', 'import time; time.sleep(1.5)'], logs=[log])
        self.assertEqual((rc, rec['verdict'], rec['stall_s']), (0, 'CLIENT_EXIT', 0.0))
        (self.root / 'watch.json').unlink()
        rc, rec, _, _, _ = self.run_watch([sys.executable, '-c', 'import time; time.sleep(600)'], logs=[log],
                                          extra=['--stall-s', '0.5'])
        self.assertEqual((rc, rec['dead']['reason']), (1, 'log-stall'))

    def test_client_ignoring_sigterm_is_killed_after_the_grace(self):
        log = self.write('engine.log', [])
        marker = self.root / 'ignoring'

        def feed():
            deadline = time.monotonic() + 30
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            log.write_bytes((FIX / 'n33-engine-stream.log').read_bytes())
        child = [sys.executable, '-c', 'import pathlib,signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                 'pathlib.Path(sys.argv[1]).touch(); time.sleep(600)', str(marker)]
        rc, rec, _, _, wall = self.run_watch(child, logs=[log], feeder=feed)
        self.assertEqual((rc, rec['dead']['reason'], rec['stop']['killed'], rec['stop']['child_rc']),
                         (1, 'engine-log', True, -signal.SIGKILL))
        self.assertLess(wall, 30)

    def test_sigterm_to_the_watch_reaches_the_client(self):
        record = self.root / 'watch.json'
        marker = self.root / 'up'
        argv = [sys.executable, '-B', str(TOOL), 'watch', '--record', str(record), '--poll-s', '0.2',
                '--gpu-cmd', self.healthy_gpus(), '--log', str(self.root / 'engine.log'), '--',
                sys.executable, '-c', 'import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(600)',
                str(marker)]
        proc = subprocess.Popen(argv)
        deadline = time.monotonic() + 30
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        proc.send_signal(signal.SIGTERM)
        self.assertEqual(proc.wait(30), 128 + signal.SIGTERM)
        rec = json.loads(record.read_text())
        self.assertEqual((rec['verdict'], rec['forwarded_signals']), ('INTERRUPTED', [signal.SIGTERM]))

    def test_stdin_reaches_the_client(self):
        out = self.root / 'stdin.txt'
        rc, _, _, _, _ = self.run_watch([sys.executable, '-c', 'import sys,pathlib; pathlib.Path(sys.argv[1]).write_text(sys.stdin.read())',
                                         str(out)], stdin='PY_HYGIENE body\n')
        self.assertEqual((rc, out.read_text()), (0, 'PY_HYGIENE body\n'))

    def test_existing_record_refuses_before_the_client_starts(self):
        (self.root / 'watch.json').write_text('{}')
        marker = self.root / 'ran'
        rc, _, _, err, _ = self.run_watch([sys.executable, '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).touch()',
                                           str(marker)])
        self.assertEqual(rc, 2)
        self.assertFalse(marker.exists())
        self.assertIn('record exists', err)


if __name__ == '__main__':
    unittest.main()
