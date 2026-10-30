"""Native28 host disk guard: evidence is committed before queue-owned stop."""
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import unittest

import e3_native28_host_observer as observer
import e3_native_performance as performance


class Native28HostGuard(unittest.TestCase):
    def test_crash_survives_failed_docker_log_capture(self):
        queue = (Path(__file__).parent / 'runs/q18a_native28.draft.sh').read_text()
        function = queue.split('host_crash_detected() {\n', 1)[1].split('\n}\n', 1)[0]
        function = 'host_crash_detected() {\n' + function + '\n}\n'
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'engine-stream.err').write_text(
                'PG Rank 0 watchdog: CUDA error: an illegal memory access was encountered\n')
            script = (function + '\nQ=' + shlex.quote(tmp) +
                      '; ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=; '
                      'host_crash_detected; printf "%s|%s" "$ENGINE_CRASHED" "$ENGINE_CRASH_SOURCE"')
            result = subprocess.run(['bash', '-c', script], capture_output=True,
                                    text=True, check=True)
            self.assertEqual(result.stdout, '1|host-engine-stream')
            (out / 'host-observer-summary.json').write_text(json.dumps({
                'shutdown': {'crash': {'kind': 'cuda_illegal_access'}}}))
            result = subprocess.run(['bash', '-c', script], capture_output=True,
                                    text=True, check=True)
            self.assertEqual(result.stdout, '1|host-observer-summary')
        scan = observer.ShutdownScan()
        scan.feed(b'2026-10-30T01:48Z DistBackendError: NCCL watchdog\n')
        self.assertEqual(scan.report()['crash']['kind'], 'nccl_dist_backend')

    def test_final_sample_and_intent_precede_queue_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            row = {'schema': 1, 'sequence': 8, 'root_free_bytes': 7*1024**3,
                   'writable_layer_bytes': 9000, 'final_root_free_guard_sample': True}
            self.assertTrue(observer.guard_requested(row))
            # Log rotation can replenish root space between trigger and final
            # inspect; the first crossing still requires a queue-owned stop.
            final = dict(row, sequence=9, root_free_bytes=9*1024**3)
            intent = observer.write_guard_intent(out, row, final)
            self.assertEqual(intent['reason'], 'root-free-space-guard')
            self.assertEqual(intent['final_sample_sequence'], 9)
            self.assertEqual(intent['trigger_root_free_bytes'], 7*1024**3)
            self.assertEqual(intent['root_free_bytes'], 9*1024**3)
            self.assertEqual(intent['final_size_rw_bytes'], 9000)
            self.assertEqual(json.loads((out/'host-stop-intent.json').read_text()), intent)
            with self.assertRaises(FileExistsError):
                observer.write_guard_intent(out, row, final)

    def test_queue_trap_runs_without_waiting_for_foreground_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp)/'stopped'
            script = ('trap "printf stopped > ' + str(marker) + '; exit 2" USR1; '
                      'sleep 30')
            proc = subprocess.Popen(['bash', '-c', script])
            try:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    children = Path(f'/proc/{proc.pid}/task/{proc.pid}/children')
                    if children.exists() and children.read_text().strip():
                        break
                    time.sleep(.01)
                start = time.monotonic()
                observer.notify_queue(proc.pid)
                proc.wait(timeout=3)
                self.assertLess(time.monotonic()-start, 3)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(marker.read_text(), 'stopped')
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=3)

    def test_read_bound_is_declared_and_does_not_change_metric(self):
        cfg = {'timeouts': {'read_s': 180.0}}
        old = performance.os.environ.get('Q18A_FIRST_ADVANCE_READ_TIMEOUT_S')
        try:
            performance.os.environ['Q18A_FIRST_ADVANCE_READ_TIMEOUT_S'] = '1200'
            self.assertEqual(performance.first_advance_read_timeout(cfg), 1200.0)
            performance.os.environ['Q18A_FIRST_ADVANCE_READ_TIMEOUT_S'] = '1201'
            with self.assertRaises(ValueError):
                performance.first_advance_read_timeout(cfg)
        finally:
            if old is None:
                performance.os.environ.pop('Q18A_FIRST_ADVANCE_READ_TIMEOUT_S', None)
            else:
                performance.os.environ['Q18A_FIRST_ADVANCE_READ_TIMEOUT_S'] = old


if __name__ == '__main__':
    unittest.main()
