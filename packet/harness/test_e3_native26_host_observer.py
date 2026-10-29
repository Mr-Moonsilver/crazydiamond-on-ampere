import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from e3_native26_host_observer import ShutdownScan, disk_sample


HERE = Path(__file__).resolve().parent


class HostObserver(unittest.TestCase):
    def test_shutdown_markers_do_not_invent_a_cause(self):
        scan = ShutdownScan()
        scan.feed(b'2026-10-30T00:01:00Z (EngineCore) [shutdown] EngineCore: tearing down local resources\n')
        scan.feed(b'2026-10-30T00:01:01Z TransportError: transport closing\n')
        result = scan.report()
        self.assertEqual(result['first_shutdown_signal']['kind'], 'executor_teardown')
        self.assertEqual(result['first_explicit_executor_teardown']['timestamp'],
                         '2026-10-30T00:01:00Z')
        self.assertFalse(result['initiating_reason_known'])
        scan.feed(b'x' * (2 * 1024 * 1024))
        self.assertLessEqual(len(scan.remainder), 1024 * 1024)

    def test_one_second_host_sampling_and_followed_log_survive_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / 'run'
            out.mkdir()
            upper = root / 'upper'
            upper.mkdir()
            (upper / 'evidence').write_bytes(b'x' * 4096)
            json_log = root / 'container-json.log'
            json_log.write_bytes(b'x' * 137)
            fake = root / 'bin'
            fake.mkdir()
            docker = fake / 'docker'
            docker.write_text(
                '#!/usr/bin/env python3\n'
                'import sys,time\n'
                "if sys.argv[1]=='inspect':\n"
                f" print('abc123|true|2026-10-30T00:00:00Z|{json_log}|{upper}')\n"
                "elif sys.argv[1]=='logs':\n"
                " print('2026-10-30T00:01:00Z [shutdown] EngineCore: tearing down local resources',flush=True)\n"
                " print('2026-10-30T00:01:01Z TransportError: transport closing',flush=True)\n"
                " time.sleep(30)\n"
                "else: raise SystemExit(2)\n")
            docker.chmod(0o700)
            env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ['PATH'],
                       PYTHONDONTWRITEBYTECODE='1')
            proc = subprocess.Popen(
                [sys.executable, '-B', str(HERE / 'e3_native26_host_observer.py'),
                 '--container', 'test-container', '--out', str(out)],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                time.sleep(2.4)
                proc.send_signal(signal.SIGTERM)
                _stdout, stderr = proc.communicate(timeout=12)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            self.assertEqual(proc.returncode, 0, stderr.decode())
            stream = (out / 'engine-stream.log').read_text()
            self.assertIn('transport closing', stream)
            rows = [json.loads(line) for line in (out / 'host-disk.jsonl').read_text().splitlines()]
            self.assertGreaterEqual(len(rows), 2)
            self.assertTrue(all(row['json_log_bytes'] == 137 for row in rows))
            self.assertTrue(all(row['writable_layer_bytes'] >= 4096 for row in rows))
            summary = json.loads((out / 'host-observer-summary.json').read_text())
            self.assertEqual(summary['shutdown']['first_shutdown_signal']['kind'],
                             'executor_teardown')
            self.assertEqual(summary['log_dropped_bytes'], 0)

    def test_queue_starts_before_restart_and_stops_after_engine(self):
        source = (HERE / 'runs/q18a_native26.draft.sh').read_text()
        self.assertLess(source.index('HOST_OBSERVER_PID=$!'),
                        source.index('systemctl restart "$SERVICE"'))
        self.assertLess(source.index('stop_engine || true'),
                        source.index('stop_host_observer || true'))
        self.assertIn('host_observation=observation', source)
        self.assertNotIn('--log-opt', source)


if __name__ == '__main__':
    unittest.main()
