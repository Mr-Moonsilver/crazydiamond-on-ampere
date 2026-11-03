#!/usr/bin/env python3
"""Native43 health-scoped engine-death scan and watch (stdlib only; no engine, Docker or GPU mutation).

Supersedes native43_engine_death.py by name (that file stays byte-identical: past queues ran it) and imports nothing
from it.  One rule differs (lead, 3 Nov 2026).  q18a-append-yield (bundle d1cd5c4bd6fd) was voided by one line, at
15:06:22Z, before the first /health 200 at 15:06:54Z: EngineCore's "No available shared memory broadcast block found
in 60 seconds" while host-KV pinning during KV init was slow.  Every startup step and the unarmed check then passed.
The stall rule did its job in l-e; it is wrong only before the engine has ever served.

STARTUP READ  A broadcast_block_stall line stamped before the first APIServer ``GET /health HTTP/1.1" 200`` line of the
       scanned logs is a startup read: recorded under startup_broadcast_stall (count, first line, wall), not a death.
       After that first health 200 it stays UNPLANNED exactly as before.  With no stamped health-200 line in the logs,
       or with no stamp on the stall line, it stays UNPLANNED too (fail-closed).  HARD markers and executor/worker
       teardown lines are unchanged at any time.  The scanned logs are one incarnation, so a restart inside them sits
       after the first health 200 and its stalls stay deaths.  The record states the rule and the health line it used.

Why the rule exists: in l-e (q18a-native43-lifecycle, 1 Nov 2026) the executor shut down at 11:03:08Z during row1.  No process
logged a cause.  The EngineCore stayed alive with dead workers, the row1 client waited on it for 25 min, and the
queue's engine-errors gate PASSed: its markers miss an executor teardown that no shutdown request preceded.

scan   Reads engine logs (``docker logs --timestamps`` lines, one engine incarnation) and decides:
       DEAD when
         - a hard death marker appears anywhere (HARD below), or
         - an executor/worker teardown line or a shared-memory broadcast stall (UNPLANNED below) appears before the
           first orderly shutdown trigger: an ORDERLY line, or the queue's host-stop-intent wall clock.
       ALIVE when no such line exists; INCONCLUSIVE when no log could be read.
       Exit 1 DEAD, 0 ALIVE, 2 INCONCLUSIVE.  The record (--out) names the first death line and every count.
watch  Runs the client command (after ``--``) and polls every --poll-s seconds.  It reports DEAD when the scan
       says DEAD, when any GPU is below --gpu-floor-mib for --gpu-polls consecutive polls, when the container is
       not running for --container-polls consecutive polls, or (only with --stall-s > 0) when no log grew for
       --stall-s.  On DEAD it sends SIGTERM to the client's process group, then SIGKILL after --kill-grace-s.
       It writes the record and exits 1.  Otherwise it exits with the client's own code, unchanged (a signal
       death becomes 128+N).  TERM/INT/HUP sent to the watch are forwarded to the client's process group.
The GPU and container commands are injectable (--gpu-cmd, --docker) so the CPU tests can drive them.
"""
import argparse
import calendar
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

SCAN_SCHEMA = 'native43-engine-death-scan-h/1'
WATCH_SCHEMA = 'native43-engine-death-watch-h/1'
TS = re.compile(rb'^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d{1,9}))?Z ')
TEXT_MAX = 320
READ_CHUNK = 16 * 1024 * 1024

# Death wherever it appears in the incarnation's log.
HARD = (
    ('worker_fatal_record', re.compile(rb'FLASH_S5_WORKER_FAILURE')),       # le-fix (b) parent-side record
    ('worker_died', re.compile(rb'Worker proc .* died unexpectedly')),      # stock monitor_workers
    ('engine_core_exited', re.compile(rb'\[shutdown\] MPClient: engine core exited unexpectedly')),
    ('engine_fatal', re.compile(rb'EngineCore encountered a fatal error')),
    ('engine_dead', re.compile(rb'EngineDeadError')),
    ('nccl_dist_backend', re.compile(rb'DistBackendError')),
    ('cuda_illegal_access', re.compile(rb'illegal memory access')),
)
# Death only when no orderly shutdown trigger precedes the line.
UNPLANNED = (
    ('executor_teardown', re.compile(rb'\[shutdown\] Executor: ')),
    ('worker_parent_exited', re.compile(rb'Parent process exited')),
    ('worker_shutdown', re.compile(rb'\[shutdown\] WorkerProc: ')),
    ('broadcast_block_stall', re.compile(rb'No available shared memory broadcast block')),
)
# A requested shutdown: the API server's trigger, the EngineCore's signal trigger or its shutdown start.
ORDERLY = (
    ('api_shutdown_triggered', re.compile(rb'\[shutdown\] API server: shutdown triggered')),
    ('engine_core_signal', re.compile(rb'\[shutdown\] EngineCore: trigger received signal')),
    ('engine_core_shutdown_start', re.compile(rb'\[shutdown\] EngineCore: start mode=')),
)
# The engine has served: the API server's access line for a 200 on /health.  Anchored to the docker stamp, the process tag
# and uvicorn's access format, so text elsewhere in a log line cannot forge it.
HEALTH_200 = re.compile(rb'^\S+ \(APIServer pid=\d+\) INFO:\s+\S+ - "GET /health HTTP/1\.1" 200\b')
STARTUP_READ_KIND = 'broadcast_block_stall'
STARTUP_RULE = ('a broadcast_block_stall line stamped before the first stamped APIServer GET /health 200 line of these logs is a '
                'startup read, not a death; after it, or with no such line, or without a stamp, it stays unplanned')
GPU_CMD = 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits'


def wall_ns(line):
    m = TS.match(line)
    if not m:
        return None
    y, mo, d, h, mi, s, frac = m.groups()
    seconds = calendar.timegm((int(y), int(mo), int(d), int(h), int(mi), int(s), 0, 0, 0))
    return seconds * 10**9 + (int((frac or b'0').ljust(9, b'0')) if frac else 0)


def iso(ns):
    if ns is None:
        return None
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(ns // 10**9)) + '.%09dZ' % (ns % 10**9)


def classify(line):
    for family, rows in (('hard', HARD), ('unplanned', UNPLANNED), ('orderly', ORDERLY)):
        for kind, rx in rows:
            if rx.search(line):
                return family, kind
    return None


def read_stop_intent(path):
    if path is None:
        return dict(path=None, state='not-given', wall_ns=None)
    p = Path(path)
    if not p.exists():
        return dict(path=str(p), state='absent', wall_ns=None)
    try:
        value = json.loads(p.read_text()).get('wall_ns')
        if type(value) is not int or value <= 0:
            raise ValueError('wall_ns')
    except (OSError, ValueError, AttributeError) as exc:
        # Fail closed: an unreadable intent bounds nothing, so a teardown still needs an orderly line.
        return dict(path=str(p), state='invalid:' + type(exc).__name__, wall_ns=None)
    return dict(path=str(p), state='read', wall_ns=value)


class Scanner:
    """Incremental reader over append-only engine logs; poll() reads what was added since the last call."""

    def __init__(self, paths, stop_intent=None):
        self.files = [dict(path=str(p), offset=0, rest=b'', lines=0, bytes=0, readable=False, error=None,
                           sha=hashlib.sha256()) for p in paths]
        self.stop_intent_path = stop_intent
        self.events = []
        self.first_health = None

    def _line(self, index, f, line):
        f['lines'] += 1
        if HEALTH_200.match(line):
            ns = wall_ns(line)
            if ns is not None and (self.first_health is None or ns < self.first_health['wall_ns']):
                self.first_health = dict(file=index, line=f['lines'], wall_ns=ns,
                                         text=line[:TEXT_MAX].decode('utf-8', 'replace'))
        hit = classify(line)
        if hit is not None:
            family, kind = hit
            self.events.append(dict(family=family, kind=kind, file=index, line=f['lines'], wall_ns=wall_ns(line),
                                    text=line[:TEXT_MAX].decode('utf-8', 'replace')))

    def poll(self, final=False):
        grown = 0
        for index, f in enumerate(self.files):
            try:
                size = os.stat(f['path']).st_size
                if size < f['offset']:
                    f['error'] = 'truncated-at-%d' % f['offset']
                    f.update(offset=0, rest=b'')
                with open(f['path'], 'rb') as handle:
                    handle.seek(f['offset'])
                    while True:
                        data = handle.read(READ_CHUNK)
                        if not data:
                            break
                        f['offset'] += len(data)
                        f['bytes'] += len(data)
                        f['sha'].update(data)
                        grown += len(data)
                        parts = (f['rest'] + data).split(b'\n')
                        f['rest'] = parts.pop()
                        for line in parts:
                            self._line(index, f, line)
                f['readable'] = True
            except FileNotFoundError:
                continue
            except OSError as exc:
                f['error'] = type(exc).__name__
                continue
            if final and f['rest']:
                self._line(index, f, f['rest'])
                f['rest'] = b''
        return grown

    def decide(self):
        intent = read_stop_intent(self.stop_intent_path)
        orderly = [e for e in self.events if e['family'] == 'orderly' and e['wall_ns'] is not None]
        first = min(orderly, key=lambda e: e['wall_ns']) if orderly else None
        bound = min([v for v in (intent['wall_ns'], first and first['wall_ns']) if v is not None], default=None)
        health = self.first_health
        deaths, startup = [], []
        for e in self.events:
            if e['family'] == 'hard':
                deaths.append(dict(e, why='hard marker'))
            elif e['family'] == 'unplanned':
                if (e['kind'] == STARTUP_READ_KIND and health is not None and e['wall_ns'] is not None
                        and e['wall_ns'] < health['wall_ns']):
                    startup.append(e)
                elif bound is None:
                    deaths.append(dict(e, why='no orderly shutdown trigger in the log and no stop intent'))
                elif e['wall_ns'] is None:
                    earlier = any(o['family'] == 'orderly' and o['file'] == e['file'] and o['line'] < e['line']
                                  for o in self.events)
                    if not earlier:
                        deaths.append(dict(e, why='untimestamped teardown, no earlier orderly line in its file'))
                elif e['wall_ns'] < bound:
                    deaths.append(dict(e, why='before the first orderly shutdown trigger'))
        readable = any(f['readable'] for f in self.files)
        verdict = 'DEAD' if deaths else ('ALIVE' if readable else 'INCONCLUSIVE')
        key = lambda e: (e['wall_ns'] if e['wall_ns'] is not None else float('inf'), e['file'], e['line'])
        first_death = min(deaths, key=key) if deaths else None
        counts = {}
        for e in deaths:
            counts[e['kind']] = counts.get(e['kind'], 0) + 1
        view = lambda e: None if e is None else dict(
            kind=e['kind'], family=e['family'], file=self.files[e['file']]['path'], line=e['line'],
            wall=iso(e['wall_ns']), text=e['text'], **({'why': e['why']} if 'why' in e else {}))
        first_startup = min(startup, key=key) if startup else None
        return dict(schema=SCAN_SCHEMA, verdict=verdict, first_death=view(first_death), death_counts=counts,
                    first_orderly=view(first), orderly_bound=iso(bound), stop_intent=intent,
                    startup_broadcast_stall=dict(count=len(startup), first=view(first_startup)),
                    first_health_200=None if health is None else dict(
                        file=self.files[health['file']]['path'], line=health['line'], wall=iso(health['wall_ns']),
                        text=health['text']),
                    files=[dict(path=f['path'], readable=f['readable'], bytes=f['bytes'], lines=f['lines'],
                                sha256=f['sha'].hexdigest() if f['readable'] else None, error=f['error'])
                           for f in self.files],
                    rules=dict(hard=[k for k, _ in HARD], unplanned_before_orderly=[k for k, _ in UNPLANNED],
                               orderly=[k for k, _ in ORDERLY] + ['stop-intent wall_ns'], startup_read=STARTUP_RULE))


def write_json(path, record, exclusive):
    if path is None:
        return
    p = Path(path)
    text = json.dumps(record, sort_keys=True, indent=1) + '\n'
    if exclusive:
        with p.open('x') as out:
            out.write(text)
    else:
        tmp = p.with_name('.%s.%d.tmp' % (p.name, os.getpid()))
        tmp.write_text(text)
        os.replace(tmp, p)


def scan(args):
    scanner = Scanner(args.log, args.stop_intent)
    scanner.poll(final=True)
    record = scanner.decide()
    write_json(args.out, record, exclusive=False)
    print('ENGINE_DEATH_SCAN %s%s' % (record['verdict'], '' if record['first_death'] is None else
          ' first=%s at %s' % (record['first_death']['kind'], record['first_death']['wall'])) +
          ('' if not record['startup_broadcast_stall']['count'] else
           ' startup_broadcast_stall=%d' % record['startup_broadcast_stall']['count']))
    return {'DEAD': 1, 'ALIVE': 0}.get(record['verdict'], 2)


def gpu_used(cmd, timeout):
    try:
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, type(exc).__name__
    if result.returncode:
        return None, 'rc%d' % result.returncode
    values = []
    for row in result.stdout.splitlines():
        if row.strip():
            try:
                values.append(int(row.split(',')[-1].strip()))
            except ValueError:
                return None, 'unparsed'
    return (values, None) if values else (None, 'empty')


def container_state(docker, name, timeout):
    try:
        result = subprocess.run(shlex.split(docker) + ['inspect', '--format', '{{.State.Running}}', name],
                                capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 'error:' + type(exc).__name__
    if result.returncode:
        return 'absent' if 'No such' in result.stderr else 'error:rc%d' % result.returncode
    return {'true': 'running', 'false': 'stopped'}.get(result.stdout.strip(), 'error:unparsed')


def _child_setup():
    os.setpgid(0, 0)
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG: the watch's death ends it
    except (OSError, AttributeError):
        pass


def stop_child(proc, grace):
    row = dict(term_wall=iso(time.time_ns()), killed=False)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        row['killed'] = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    row['child_rc'] = proc.returncode
    return row


def watch(args, child):
    if not child:
        print('ENGINE_DEATH_WATCH usage: a client command after --', file=sys.stderr)
        return 2
    if args.record and Path(args.record).exists():
        print('ENGINE_DEATH_WATCH refused: record exists ' + args.record, file=sys.stderr)
        return 2
    import ctypes  # noqa: F401  imported before fork for _child_setup
    scanner = Scanner(args.log, args.stop_intent)
    record = dict(schema=WATCH_SCHEMA, client=[os.path.basename(child[0])] + child[1:3], started=iso(time.time_ns()),
                  poll_s=args.poll_s, gpu_floor_mib=args.gpu_floor_mib, gpu_polls=args.gpu_polls,
                  container=args.container, container_polls=args.container_polls, stall_s=args.stall_s,
                  logs=list(args.log), polls=0, gpu_errors=0, container_errors=0, last_gpu=None, last_container=None,
                  startup_read=STARTUP_RULE, startup_broadcast_stall=None, first_health_200=None)
    proc = subprocess.Popen(child, preexec_fn=_child_setup)
    record['client_pid'] = proc.pid
    forwarded = []

    def forward(signum, _frame):
        forwarded.append(signum)
        try:
            os.killpg(proc.pid, signum)
        except ProcessLookupError:
            pass
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)
    low = down = 0
    last_growth = time.monotonic()
    dead = None
    while dead is None:
        if proc.poll() is not None:
            break
        if forwarded:
            try:
                proc.wait(timeout=args.kill_grace_s)
            except subprocess.TimeoutExpired:
                record['interrupt_kill'] = stop_child(proc, 5)
            break
        record['polls'] += 1
        if scanner.poll():
            last_growth = time.monotonic()
        decision = scanner.decide()
        record.update(startup_broadcast_stall=decision['startup_broadcast_stall'],
                      first_health_200=decision['first_health_200'])
        if decision['verdict'] == 'DEAD':
            dead = dict(reason='engine-log', first_death=decision['first_death'], death_counts=decision['death_counts'])
            break
        used, err = gpu_used(args.gpu_cmd, args.cmd_timeout_s)
        if used is None:
            record['gpu_errors'] += 1
        else:
            record['last_gpu'] = used
            low = low + 1 if any(v < args.gpu_floor_mib for v in used) else 0
            if low >= args.gpu_polls:
                dead = dict(reason='gpu-below-floor', gpu_used_mib=used, consecutive_polls=low)
                break
        if args.container:
            state = container_state(args.docker, args.container, args.cmd_timeout_s)
            record['last_container'] = state
            if state.startswith('error'):
                record['container_errors'] += 1
            else:
                down = down + 1 if state in ('stopped', 'absent') else 0
                if down >= args.container_polls:
                    dead = dict(reason='container-not-running', state=state, consecutive_polls=down)
                    break
        if args.stall_s > 0 and time.monotonic() - last_growth > args.stall_s:
            dead = dict(reason='log-stall', stalled_s=round(time.monotonic() - last_growth, 1))
            break
        try:
            proc.wait(timeout=args.poll_s)
        except subprocess.TimeoutExpired:
            pass
    if dead is not None:
        dead['detected'] = iso(time.time_ns())
        record.update(verdict='DEAD', dead=dead, stop=stop_child(proc, args.kill_grace_s))
        rc = 1
    else:
        rc = proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
        record.update(verdict='INTERRUPTED' if forwarded else 'CLIENT_EXIT', client_rc=proc.returncode,
                      forwarded_signals=forwarded)
    record['finished'] = iso(time.time_ns())
    try:
        write_json(args.record, record, exclusive=True)
    except OSError as exc:
        print('ENGINE_DEATH_WATCH record not written: ' + type(exc).__name__, file=sys.stderr)
    if dead is not None:
        print('ENGINE_DEATH_WATCH DEAD reason=%s; client stopped (%s)' % (
            dead['reason'], 'SIGKILL' if record['stop']['killed'] else 'SIGTERM'), file=sys.stderr)
    return rc


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    child = []
    if '--' in argv:
        cut = argv.index('--')
        argv, child = argv[:cut], argv[cut + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)
    for name in ('scan', 'watch'):
        s = sub.add_parser(name)
        s.add_argument('--log', action='append', required=True, help='engine log file; repeat for each')
        s.add_argument('--stop-intent', help="the queue's host-stop-intent.json (wall_ns); absent is fine")
    s = sub.choices['scan']
    s.add_argument('--out')
    w = sub.choices['watch']
    w.add_argument('--record', required=True)
    w.add_argument('--poll-s', type=float, default=5.0)
    w.add_argument('--gpu-cmd', default=GPU_CMD)
    w.add_argument('--gpu-floor-mib', type=int, default=1024)
    w.add_argument('--gpu-polls', type=int, default=2)
    w.add_argument('--container')
    w.add_argument('--docker', default='docker')
    w.add_argument('--container-polls', type=int, default=2)
    w.add_argument('--stall-s', type=float, default=0.0, help='0 (default) disables the log-stall rule')
    w.add_argument('--kill-grace-s', type=float, default=15.0)
    w.add_argument('--cmd-timeout-s', type=float, default=10.0)
    args = p.parse_args(argv)
    if args.mode == 'scan':
        if child:
            p.error('scan takes no client command')
        return scan(args)
    return watch(args, child)


if __name__ == '__main__':
    raise SystemExit(main())
