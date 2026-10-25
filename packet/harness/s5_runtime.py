"""CPU-only S5 progress and watchdog. No engine/queue control.

The Python alarm raises Inconclusive where Python can regain control. A separate
process writes the same failure evidence and kills ONLY its parent test rank if
CUDA holds the GIL beyond the grace period. Closing stdin disarms that process.
"""
import faulthandler
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
import time
from s5_clock import Inconclusive


class Progress:
    def __init__(self, output, rank):
        self.path = Path(str(output) + f'.rank{rank}.progress.json')
        self.events = Path(str(output) + f'.rank{rank}.progress.jsonl')
        self.lock = threading.Lock()
        self.state = dict(rank=rank, pid=os.getpid(), phase='initializing', lanes={})
        self.seq = 0

    def update(self, phase=None, **fields):
        with self.lock:
            if phase is not None:
                self.state['phase'] = phase
            self.state.update(fields, monotonic_s=time.monotonic(), wall_s=time.time())
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(self.state, allow_nan=False) + '\n')
            os.replace(tmp, self.path)
            self._event(dict(kind='phase', **self.state))

    def _event(self, row):
        self.seq += 1
        row = dict(row, seq=self.seq, monotonic_ns=time.monotonic_ns(), rank=self.state['rank'])
        # Append rather than rewrite history. Ordering is host issue order, NOT
        # GPU execution order, and this instrumentation has measurable overhead.
        with self.events.open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def event(self, lane, stage, **fields):
        with self.lock:
            self._event(dict(kind='lane', phase=self.state['phase'], lane=lane, stage=stage, **fields))


def failure_record(progress, reason):
    return dict(fatal_correctness=False, grade={'verdict':'INCONCLUSIVE'},
                error=reason, partial_rank=progress, watchdog=True)


class Watchdog:
    def __init__(self, progress, output, rank, grace=5.):
        self.progress, self.timeout = progress, None
        self.failure = Path(str(output) + f'.rank{rank}.error.json')
        self.old_handler = signal.signal(signal.SIGALRM, self._alarm)
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        self.child = subprocess.Popen([sys.executable, __file__, '--watch',
            str(os.getpid()), str(progress.path), str(self.failure)], stdin=subprocess.PIPE, text=True)
        self.grace = grace

    def _alarm(self, *_):
        faulthandler.dump_traceback(all_threads=True)
        raise Inconclusive('S5 phase watchdog expired', dict(self.progress.state, timeout_s=self.timeout))

    def arm(self, seconds, phase):
        signal.setitimer(signal.ITIMER_REAL, 0)
        self.progress.update(phase, timeout_s=seconds)
        self.timeout = seconds
        self.child.stdin.write(json.dumps({'deadline':time.monotonic()+seconds+self.grace})+'\n')
        self.child.stdin.flush()
        signal.setitimer(signal.ITIMER_REAL, seconds)

    def close(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old_handler)
        self.child.stdin.close()
        self.child.wait(timeout=3)


def watch(parent, progress_path, failure_path):
    deadline = None
    # Unbuffered reads avoid select overlooking a second already-buffered line.
    pending = b''
    while os.getppid() == parent:
        wait = None if deadline is None else max(0, deadline-time.monotonic())
        ready, _, _ = select.select([0], [], [], wait)
        if ready:
            data = os.read(0, 4096)
            if not data:
                return
            pending += data
            while b'\n' in pending:
                line, pending = pending.split(b'\n', 1)
                deadline = json.loads(line)['deadline']
            continue
        try:
            progress = json.loads(Path(progress_path).read_text())
        except (OSError, ValueError):
            progress = {'pid':parent, 'phase':'unknown'}
        try:
            with open(failure_path, 'x') as f:
                json.dump(failure_record(progress, 'S5 hard watchdog: rank unresponsive'), f)
                f.write('\n')
        except FileExistsError:
            pass
        if os.getppid() == parent:
            os.kill(parent, signal.SIGUSR1)  # C-level faulthandler, even if GIL held
            time.sleep(.5)
        if os.getppid() == parent:
            os.kill(parent, signal.SIGKILL)
        return


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] != '--watch':
        raise SystemExit('internal watchdog only')
    watch(int(sys.argv[2]), sys.argv[3], sys.argv[4])
