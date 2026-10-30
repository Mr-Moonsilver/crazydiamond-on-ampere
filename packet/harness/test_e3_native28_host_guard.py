"""Native28 host disk guard: evidence is committed before queue-owned stop."""
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

import e3_native28_host_observer as observer
import e3_native_performance as performance


class Native28HostGuard(unittest.TestCase):
    def test_final_sample_and_intent_precede_queue_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            row = {'schema': 1, 'sequence': 8, 'root_free_bytes': 7*1024**3,
                   'writable_layer_bytes': 9000, 'final_root_free_guard_sample': True}
            self.assertTrue(observer.guard_requested(row))
            intent = observer.write_guard_intent(out, row)
            self.assertEqual(intent['reason'], 'root-free-space-guard')
            self.assertEqual(intent['final_sample_sequence'], 8)
            self.assertEqual(intent['final_size_rw_bytes'], 9000)
            self.assertEqual(json.loads((out/'host-stop-intent.json').read_text()), intent)
            with self.assertRaises(FileExistsError):
                observer.write_guard_intent(out, row)

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
