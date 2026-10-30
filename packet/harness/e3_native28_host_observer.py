#!/usr/bin/env python3
"""Queue-owned, host-only Docker log and disk observer for native28.

No Docker or engine mutation is performed. A bounded in-memory tail survives a
brief ENOSPC. Below the fixed root-free threshold, the observer saves one final
Docker-size sample and asks its owning queue process to run normal cleanup.
"""

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time


TAIL_LIMIT = 32 * 1024 * 1024
ROOT_FREE_GUARD_BYTES = 8 * 1024**3
STOP = False
SHUTDOWN_MARKERS = (
    ('executor_teardown', re.compile(r'\[shutdown\] EngineCore: tearing down local resources|executor shutdown', re.I)),
    ('worker_failure', re.compile(r'WorkerProc.*(?:died|failed|exception)|worker_failure', re.I)),
    ('transport_closing', re.compile(r'TransportError: transport closing|transport closing', re.I)),
    ('engine_fatal', re.compile(r'EngineCore encountered a fatal error', re.I)),
    ('api_shutdown', re.compile(r'INFO: +Shutting down', re.I)),
)


def stop(_signum, _frame):
    global STOP
    STOP = True


def inspect(container):
    result = subprocess.run(
        ['docker', 'inspect', '--format',
         '{{.Id}}|{{.State.Running}}|{{.State.StartedAt}}|{{.LogPath}}',
         container], capture_output=True, text=True, timeout=2, check=False)
    if result.returncode:
        return None
    fields = result.stdout.strip().split('|')
    if len(fields) != 4 or fields[1] != 'true':
        return None
    return dict(container_id=fields[0], started_at=fields[2],
                log_path=fields[3], name=container)


def disk_sample(identity, sequence):
    root = os.statvfs('/')
    row = dict(schema=1, sequence=sequence, wall_ns=time.time_ns(),
               root_free_bytes=root.f_bavail * root.f_frsize,
               root_total_bytes=root.f_blocks * root.f_frsize,
               container_id=identity['container_id'][:12],
               json_log_bytes=None, writable_layer_bytes=None, errors=[])
    try:
        row['json_log_bytes'] = os.stat(identity['log_path']).st_size
    except (OSError, ValueError) as exc:
        row['errors'].append('log_stat:' + type(exc).__name__)
    try:
        # Docker's SizeRw is the same writable-layer accounting used by
        # `docker ps -s`; current Docker no longer exposes GraphDriver.UpperDir.
        result = subprocess.run(['docker', 'inspect', '--size', '--format',
                                 '{{.SizeRw}}', identity['name']],
                                capture_output=True, text=True, timeout=0.8,
                                check=False)
        if result.returncode:
            row['errors'].append('size_rw:rc' + str(result.returncode))
        else:
            row['writable_layer_bytes'] = int(result.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        row['errors'].append('size_rw:' + type(exc).__name__)
    return row


class Sink:
    """Append-only output plus bounded RAM tail during transient ENOSPC."""

    def __init__(self, path, limit=TAIL_LIMIT):
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
        self.pending = deque()
        self.pending_bytes = 0
        self.dropped_bytes = 0
        self.limit = limit
        self.write_errors = 0
        self.next_retry = 0.0

    def add(self, data):
        if not data:
            return
        self.pending.append(data)
        self.pending_bytes += len(data)
        while self.pending_bytes > self.limit and self.pending:
            removed = self.pending.popleft()
            self.pending_bytes -= len(removed)
            self.dropped_bytes += len(removed)
        self.flush()

    def flush(self, *, force=False):
        if not force and time.monotonic() < self.next_retry:
            return
        while self.pending:
            data = self.pending[0]
            try:
                written = os.write(self.fd, data)
            except OSError:
                self.write_errors += 1
                self.next_retry = time.monotonic() + 1.0
                break
            if written == len(data):
                self.pending.popleft()
                self.pending_bytes -= written
            elif written:
                self.pending[0] = data[written:]
                self.pending_bytes -= written
            else:
                self.write_errors += 1
                self.next_retry = time.monotonic() + 1.0
                break

    def close(self):
        try:
            self.flush(force=True)
            os.fsync(self.fd)
        finally:
            os.close(self.fd)


class ShutdownScan:
    def __init__(self):
        self.remainder = b''
        self.first_by_kind = {}

    def feed(self, data):
        lines = (self.remainder + data).split(b'\n')
        self.remainder = lines.pop()
        if len(self.remainder) > 1024 * 1024:
            # A malformed or enormous log line must not grow observer memory.
            self.remainder = self.remainder[-1024 * 1024:]
        for raw in lines:
            line = raw.decode('utf-8', 'replace')
            timestamp = line.split(' ', 1)[0]
            for kind, pattern in SHUTDOWN_MARKERS:
                if kind not in self.first_by_kind and pattern.search(line):
                    self.first_by_kind[kind] = dict(kind=kind,
                                                    timestamp=timestamp,
                                                    observed_wall_ns=time.time_ns())

    def report(self):
        events = list(self.first_by_kind.values())
        first = min(events, key=lambda x: x['observed_wall_ns']) if events else None
        explicit = self.first_by_kind.get('executor_teardown')
        return dict(schema=1, first_shutdown_signal=first,
                    first_explicit_executor_teardown=explicit,
                    initiating_reason_known=False,
                    note='Stream markers show observed order, not the caller or reason that initiated executor.shutdown.')


def guard_requested(row, threshold=ROOT_FREE_GUARD_BYTES):
    return type(row.get('root_free_bytes')) is int and row['root_free_bytes'] < threshold


def write_guard_intent(out, final_row, *, threshold=ROOT_FREE_GUARD_BYTES):
    """Commit the evidence pointer before the queue receives its stop signal."""
    assert guard_requested(final_row, threshold)
    canonical = json.dumps(final_row, sort_keys=True, separators=(',', ':')).encode()
    intent = dict(schema=1, reason='root-free-space-guard',
                  scope='host evidence first; queue-owned engine stop requested',
                  wall_ns=time.time_ns(), threshold_bytes=threshold,
                  root_free_bytes=final_row['root_free_bytes'],
                  final_sample_sequence=final_row['sequence'],
                  final_sample_sha256=hashlib.sha256(canonical).hexdigest(),
                  final_size_rw_bytes=final_row['writable_layer_bytes'])
    fd = os.open(out / 'host-stop-intent.json',
                 os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(intent, handle, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return intent


def notify_queue(queue_pid):
    """Interrupt only pre-existing queue children so Bash can run its trap now."""
    children_path = Path('/proc') / str(queue_pid) / 'task' / str(queue_pid) / 'children'
    try:
        children = [int(value) for value in children_path.read_text().split()]
    except (OSError, ValueError):
        children = []
    os.kill(queue_pid, signal.SIGUSR1)
    for pid in children:
        if pid == os.getpid():
            continue
        try:
            # Do not touch a new cleanup child started by the queue trap.
            status = (Path('/proc') / str(pid) / 'status').read_text()
            parent = next((line.split(':', 1)[1].strip() for line in status.splitlines()
                           if line.startswith('PPid:')), None)
            if parent != str(queue_pid):
                continue
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            continue


def watch(container, out, exclude_id='', interval=1.0, *, queue_pid=None,
          threshold=ROOT_FREE_GUARD_BYTES):
    out = Path(out)
    if not out.is_dir() or out.is_symlink():
        raise ValueError('existing regular run directory required')
    logs = Sink(out / 'engine-stream.log')
    samples = Sink(out / 'host-disk.jsonl')
    scan = ShutdownScan()
    identity = None
    process = None
    stderr = None
    sequence = 0
    guard = None
    next_sample = time.monotonic()
    started = time.monotonic()
    try:
        while not STOP:
            if identity is None:
                candidate = inspect(container)
                if candidate and candidate['container_id'] != exclude_id:
                    identity = candidate
                    stderr = open(out / 'engine-stream.err', 'xb')
                    process = subprocess.Popen(
                        ['docker', 'logs', '--follow', '--timestamps',
                         '--since', identity['started_at'], container],
                        stdout=subprocess.PIPE, stderr=stderr, bufsize=0)
                    selector = selectors.DefaultSelector()
                    selector.register(process.stdout, selectors.EVENT_READ)
                elif time.monotonic() - started > 360:
                    break
            now = time.monotonic()
            if identity and now >= next_sample:
                row = disk_sample(identity, sequence)
                samples.add((json.dumps(row, sort_keys=True) + '\n').encode())
                sequence += 1
                next_sample += interval
                if next_sample < now:
                    next_sample = now + interval
                if guard is None and guard_requested(row, threshold):
                    # The first crossing is descriptive; take a final fresh
                    # statvfs + docker inspect --size sample before requesting
                    # queue-owned stop. No engine control occurs here.
                    final = disk_sample(identity, sequence)
                    final['final_root_free_guard_sample'] = True
                    samples.add((json.dumps(final, sort_keys=True) + '\n').encode())
                    sequence += 1
                    samples.flush(force=True)
                    os.fsync(samples.fd)
                    guard = write_guard_intent(out, final, threshold=threshold)
                    if queue_pid is not None:
                        notify_queue(queue_pid)
                    break
            if process:
                for key, _ in selector.select(timeout=0.1):
                    chunk = os.read(key.fd, 65536)
                    if chunk:
                        logs.add(chunk)
                        scan.feed(chunk)
                    else:
                        selector.unregister(key.fileobj)
                if process.poll() is not None and not selector.get_map():
                    # Docker's logger can exit on ENOSPC while the container
                    # remains alive. Continue the host samples until cleanup.
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if process and process.stdout:
            for chunk in iter(lambda: process.stdout.read(65536), b''):
                logs.add(chunk)
                scan.feed(chunk)
    finally:
        for sink in (logs, samples):
            try:
                sink.close()
            except OSError:
                pass
        if stderr:
            stderr.close()
        summary = dict(schema=1, container_id=None if identity is None else identity['container_id'][:12],
                       samples=sequence, docker_logs_rc=None if process is None else process.returncode,
                       log_pending_bytes=logs.pending_bytes,
                       log_dropped_bytes=logs.dropped_bytes,
                       log_write_errors=logs.write_errors,
                       sample_pending_bytes=samples.pending_bytes,
                       sample_dropped_bytes=samples.dropped_bytes,
                       sample_write_errors=samples.write_errors,
                       shutdown=scan.report())
        summary['root_free_guard'] = guard
        try:
            with (out / 'host-observer-summary.json').open('x') as handle:
                json.dump(summary, handle, sort_keys=True)
                handle.write('\n')
        except OSError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--exclude-id', default='')
    parser.add_argument('--queue-pid', type=int, required=True)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.queue_pid <= 1:
        raise ValueError('queue pid required')
    watch(args.container, args.out, args.exclude_id, queue_pid=args.queue_pid)


if __name__ == '__main__':
    main()
