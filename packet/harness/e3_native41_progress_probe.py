#!/usr/bin/env python3
"""Held-f connected CPU probe: native progress receipts and the burst candidate.

Same composition as e3_native41_auto_probe (production install_engine, real
EngineCore.step_with_batch_queue, scheduler, controller, driver, worker bridges
and dispatchers, the auto-arm policy). The setting reaches install_engine
through its env argument. Modes: unset, trace, trace-skew, burst, burst-skew
(the -skew modes add the tp-skew asynchronous per-rank completion leaf) and
burst-current (burst installed, the arm switched to mode current).

Fixture leaves, beyond native41_fixture_leaves.json:

* progress-meta-pop: DirectTransport calls each worker method directly. The
  generated ``WorkerProc._execute_worker_rpc`` pops the reserved kwarg through
  ``executor_hooks.worker_rpc_begin`` before dispatch; this leaf calls that same
  real hook on a per-rank copy of the kwargs, so a worker method never sees it.
* direct-flag-record: DirectTransport has no receiver threads, so the
  ``direct`` registration flag is recorded and checked against the worker-side
  meta bit (the two-FIFO receiver itself is proven in test_e3_native_reply_hol).
* sample-wait: the probe's sample future stands for ``StockLocalFuture``. Its
  ``result`` applies the same gate (sample future, ``wait_allowed``) and runs the
  real ``executor_hooks.native_progress_wait``, with ``ready`` turning true after
  WAIT_ROUNDS rounds (it stands for the decode batch's completion). The execute
  and sample fixtures set ``last_stock_kind`` as ``collective_rpc`` does while
  recording.
* DirectTransport carries no ``registered_s``/lag and no stock RPC goes through
  ``collective_rpc``: receive times and 's'/'w' rows are absent here by
  construction (proven over the generated worker loop in test_e3_native_reply_hol).
"""
import argparse
import contextlib
import hashlib
import json
import logging
from pathlib import Path
import traceback
from types import SimpleNamespace as NS
from unittest import mock

import e3_native41_auto_probe as ap
from e3_native40_unarmed_probe import overlay_aliases

MODES = ('unset', 'trace', 'trace-skew', 'burst', 'burst-skew', 'burst-current')
WAIT_ROUNDS = 5
WAIT_SEGMENTS = ('prefix', 'layer', 'suffix')


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith('flash-s5 progress '):
            self.lines.append(message)


class PopMeta:
    """Leaf progress-meta-pop (see module docstring)."""

    def __init__(self, worker, rank, hooks, log):
        self._worker, self._rank, self._hooks, self._log = worker, rank, hooks, log

    def __getattr__(self, name):
        real = getattr(self._worker, name)
        if not callable(real):
            return real

        def call(*args, **kwargs):
            local = dict(kwargs)
            rpc = self._hooks.worker_rpc_begin(NS(rank=self._rank), name, local)
            if rpc is not None:
                self._log.append((self._rank, rpc[0], name, rpc[4]))
            return real(*args, **local)
        return call


class SampleWait:
    """Leaf sample-wait (see module docstring)."""

    def __init__(self, inner, progress, hooks, waits):
        self.inner, self.progress, self.hooks, self.waits = inner, progress, hooks, waits

    def done(self):
        return self.inner.done()

    def result(self, timeout=None):
        progress = self.progress
        if progress is not None and progress.wait_allowed():
            rounds = [0]

            def ready():
                rounds[0] += 1
                return rounds[0] > WAIT_ROUNDS
            self.waits.append(self.hooks.native_progress_wait(progress, ready, None))
        return self.inner.result(timeout)


def connected(setting):
    import s5_startup as startup
    real = startup.install_engine

    def install(engine, *, env=None, **kwargs):
        env = dict(env or {})
        if setting is not None:
            env['FLASH_S5_NATIVE_PROGRESS'] = setting
        return real(engine, env=env, **kwargs)
    with mock.patch.object(startup, 'install_engine', install):
        conn = ap.Connected('1')
    return conn


def set_mode(conn, mode):
    """The box arm's control sequence: pause (lease), native_progress, release, resume."""
    from vllm.v1.core.sched.interface import PauseState
    scheduler = conn.scheduler
    old = scheduler.pause_state
    scheduler.set_pause_state(PauseState.PAUSED_ALL)
    try:
        reply = conn.driver.submit_control(dict(action='native_progress', mode=mode,
                                                engine_epoch=conn.driver.engine_epoch))
        leased = conn.driver.auto_arm.manual_owned
        release = conn.driver.submit_control(dict(action='release_auto_arm',
                                                  engine_epoch=conn.driver.engine_epoch))
    finally:
        scheduler.set_pause_state(old)
    return reply, leased, release


def run(mode):
    # Overlay modules are aliased first, exactly as e3_native41_auto_probe.run.
    root, failures = overlay_aliases()
    import engine_driver as ed
    import executor_hooks as eh
    import native_measurement as nm
    setting = None if mode == 'unset' else mode.split('-')[0]
    skew = mode.endswith('-skew')
    selected = 'burst' if mode in ('burst', 'burst-skew') else 'current'
    conn = connected(setting)
    conn.dispatched = set()
    popped, flags, waits, violations = [], {}, [], []
    conn.transport.network = NS(workers={rank: PopMeta(w, rank, eh, popped)
                                         for rank, w in conn.workers.items()})
    driver = conn.driver
    real_submit = conn.transport.submit

    def submit(kind, send, direct=False, **kwargs):
        sequence = real_submit(kind, send, **kwargs)
        flags[sequence] = (kind, direct)
        return sequence
    conn.transport.submit = submit

    def recording():
        return getattr(conn.executor, '_flash_s5_progress_recording', None)

    def note(kind):
        progress = recording()
        if progress is not None:
            progress.last_stock_kind = kind
    real_execute = conn.executor.execute_model

    def execute(out, **kwargs):
        note('execute_model')
        return real_execute(out, **kwargs)
    conn.executor.execute_model = execute

    def sample(*args, **kwargs):
        progress = recording()
        note('sample_tokens')
        return SampleWait(conn.future, progress, eh, waits)
    conn.executor.sample_tokens = sample

    # In-wait guard: completion-dependent and scheduler-visible driver actions
    # never run inside an owner wait; only permitted kinds are sent there.
    def forbid(owner, name):
        real = getattr(owner, name)

        def call(*args, **kwargs):
            if driver._in_wait:
                violations.append(name)
            return real(*args, **kwargs)
        setattr(owner, name, call)
    for owner, name in ((driver.controller, 'bind'), (driver.controller, 'note_pp_send'),
                        (driver.controller, 'note_worker_release'), (driver, '_send_release'),
                        (driver, '_begin'), (driver, '_reset'), (driver, '_next_receipt'),
                        (driver, 'pump_control'), (driver, '_drive_route'), (driver, '_drive_stop')):
        forbid(owner, name)
    real_observed = driver._submit_observed

    def observed(kind, envelope, ranks, **kwargs):
        if driver._in_wait:
            command = driver._command
            if not (kind == 'profile' or (kind == 'segment' and command is not None
                                          and command.kind.value in WAIT_SEGMENTS)):
                violations.append('wait-sent-%s-%s' % (
                    kind, getattr(getattr(command, 'kind', None), 'value', None)))
        return real_observed(kind, envelope, ranks, **kwargs)
    driver._submit_observed = observed

    permits, sampled = [], []

    def state():
        command = driver._retry
        if command is None and driver.index < len(driver.plan):
            upcoming = driver.plan[driver.index]['kind'].value
        else:
            upcoming = None if command is None else command.kind.value
        inflight = tuple(driver._inflight.values())
        current = (driver._command.kind.value if inflight == ('segment',)
                   and driver._command is not None else None)
        return (driver.phase, inflight, current, upcoming)
    if setting is not None:
        real_wait = driver.progress_in_wait

        def progress_in_wait():
            before = state()
            out = real_wait()
            permits.append((before, out))
            return out
        driver.progress_in_wait = progress_in_wait
        real_advance = driver.advance

        def advance(*args, **kwargs):
            out = real_advance(*args, **kwargs)
            # The permit is side-effect free: sampled at every tick it covers
            # every state the chunk passes through, not only where waits land.
            if driver.progress.recording:
                sampled.append((state(), driver._wait_permit()))
            return out
        driver.advance = advance
    mode_reply = None
    if setting == 'burst':
        mode_reply = set_mode(conn, selected)
    capture = Capture()
    # The modules' own loggers (vllm.v1.flash_e3.* under the overlay aliases).
    loggers = [ed._progress_logger(), eh._progress_logger()]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(capture)
        lg.setLevel(logging.INFO)
    try:
        with contextlib.ExitStack() as scope:
            if skew:
                conn.skew = scope.enter_context(ap.Skew(conn).installed())
            result = ap._run('tp-skew' if skew else 'auto', conn, root, failures)
    finally:
        for lg, level in zip(loggers, levels):
            lg.removeHandler(capture)
            lg.setLevel(level)
    checks = result['checks']

    def check(name, ok, **details):
        checks.append(dict(name=name, ok=bool(ok), **details))
        assert ok, name + ' ' + json.dumps(details, sort_keys=True, default=str)[:2000]
    check('no-driver-action-inside-a-wait', not violations, violations=violations[:16])
    progress = driver.progress
    if setting is None:
        check('unset-no-progress', progress is None and not capture.lines and not popped
              and not waits and all(not d for _k, d in flags.values())
              and recording() is None)
        result.update(progress_lines=0)
        return result
    burst = selected == 'burst'
    if setting == 'burst':
        reply, leased, release = mode_reply
        check('mode-switch-through-the-arm-sequence',
              reply['mode'] == selected and reply['previous'] == 'current'
              and reply['installed'] == 'burst' and leased is True
              and release.get('released') is True and not driver.auto_arm.manual_owned,
              reply={k: reply[k] for k in ('mode', 'previous', 'installed', 'route')})
    lines = capture.lines
    check('progress-lines-bounded', lines and all(len(t) < nm.PROGRESS_LINE_BYTES for t in lines),
          longest=max(map(len, lines)) if lines else 0)
    bodies = [json.loads(t[len(nm.PROGRESS_LINE):]) for t in lines]
    engine = [b for b in bodies if b['side'] == 'engine']
    check('only-engine-lines-in-process', len(engine) == len(bodies))
    chunks = {}
    for body in engine:
        chunks.setdefault(body['line'], []).append(body)
    joined = []
    for number in sorted(chunks):
        parts = sorted(chunks[number], key=lambda b: b['part'])
        check('parts-complete-%d' % number,
              [b['part'] for b in parts] == list(range(parts[0]['parts']))
              and len({(b['txn'], b['base_ns'], b['parts']) for b in parts}) == 1
              and parts[0]['dropped'] == 0
              and parts[0]['fields'] == {k: list(v) for k, v in nm.PROGRESS_ROWS.items()}
              and all('fields' not in b for b in parts[1:]))
        rows = [r for b in parts for r in b['rows'].get('n', [])]
        joined.append((parts[0], rows))
    native = conn.native
    check('one-flush-per-native-chunk', len(joined) == len(native)
          and [(h['chunk_start'], h['chunk_end']) for h, _ in joined]
          == [(r['start'], r['end']) for r in native],
          flushes=len(joined), chunks=len(native))
    at = nm.PROGRESS_N
    seqs, wait_rows = [], []
    for head, rows in joined:
        kinds = [r[at['kind']] for r in rows]
        seqs += [r[at['seq']] for r in rows]
        wait_rows += [r for r in rows if r[at['origin']] == 'wait']
        check('chunk-%d-command-flow' % head['txn'],
              kinds and kinds[0] == 'open' and kinds[-1] == 'release' and 'segment' in kinds
              and head['open_rows'] == 0 and head['mode'] == ('burst' if burst else 'current')
              and type(head['decode_live']) is int and head['decode_live'] >= 1,
              kinds=sorted(set(kinds)), decode_live=head['decode_live'])
        check('chunk-%d-stamps-ordered' % head['txn'], all(
            r[at['t_submit']] <= r[at['t_sent']] <= r[at['t_apply']]
            and r[at['outcome']] is not None
            and r[at['direct']] == int(burst and r[at['kind']] in ('segment', 'profile', 'poll'))
            for r in rows))
        check('chunk-%d-segment-identity' % head['txn'], all(
            type(r[at['op_seq']]) is int and r[at['stage']] in (0, 1) and r[at['segment']] is not None
            for r in rows if r[at['kind']] == 'segment'))
    sent = sorted({seq for rank, seq, name, direct in popped if rank == 0})
    meta_direct = {seq: direct for rank, seq, name, direct in popped if rank == 0}
    check('every-recorded-command-carried-meta-and-every-meta-was-recorded',
          sorted(seqs) == sent and len(set(seqs)) == len(seqs)
          and {rank for rank, *_ in popped} == set(conn.workers),
          recorded=len(seqs), sent=len(sent))
    check('direct-registration-equals-the-worker-meta-bit',
          all(flags[seq][1] == meta_direct[seq] for seq in sent)
          and all(not d for seq, (k, d) in flags.items() if seq not in meta_direct),
          direct=sum(meta_direct.values()))
    snapshot = driver._status_record()['native_progress']
    check('status-receipt', snapshot['installed'] == setting
          and snapshot['mode'] == ('burst' if burst else 'current')
          and snapshot['chunks'] == len(native) and snapshot['line_errors'] == 0
          and snapshot['commands'] == len(seqs)
          and snapshot['direct_commands'] == sum(meta_direct.values())
          and snapshot['trace_dropped'] == 0 and not snapshot['recording'], snapshot=snapshot)
    if burst:
        check('burst-progressed-inside-waits',
              waits and sum(s for s, _c in waits) > 0 and wait_rows
              and snapshot['wait_sends'] == sum(s for s, _c in waits) == len(wait_rows)
              and snapshot['wait_capped'] == sum(c for _s, c in waits),
              waits=len(waits), wait_sends=snapshot['wait_sends'], wait_rows=len(wait_rows))
        def permitted(state):
            phase, inflight, current, upcoming = state
            if phase == 'profile':
                return inflight in ((), ('profile',))
            return phase == 'segments' and (current in WAIT_SEGMENTS if inflight
                                            else upcoming in WAIT_SEGMENTS)
        granted = [st for st, out in permits if out is not None]
        check('wait-permit-granted-only-in-its-states',
              granted and all(permitted(st) for st in granted),
              granted=len(granted), refused=len(permits) - len(granted))
        refused = [st for st, ok in sampled if not ok]
        # A segments state is named by its in-flight command, else the next one.
        refused_kinds = ({st[0] for st in refused if st[0] != 'segments'}
                         | {(st[2] if st[1] else st[3]) for st in refused if st[0] == 'segments'})
        check('tick-sampled-permit-true-only-in-its-states-and-false-at-every-tick-state',
              all(permitted(st) for st, ok in sampled if ok)
              and {'open', 'bind', 'finalize', 'release'} <= refused_kinds
              and {'prepare', 'pp_send', 'pp_receive', 'state_finalize'} <= refused_kinds,
              sampled=len(sampled), refused=len(refused), refused_kinds=sorted(map(str, refused_kinds)))
        check('wait-sent-only-in-chunk-segments-and-profile-queries', all(
            (r[at['kind']] == 'segment' and r[at['segment']] in WAIT_SEGMENTS)
            or r[at['kind']] == 'profile' for r in wait_rows),
            kinds=sorted({(r[at['kind']], r[at['segment']]) for r in wait_rows}))
    else:
        check('current-never-progresses-in-a-wait', not waits and not wait_rows and not permits
              and snapshot['wait_sends'] == 0)
    check('not-recording-after-cycle', recording() is None)
    result.update(progress_lines=len(lines), progress_rows=len(seqs), wait_rows=len(wait_rows),
                  waits=len(waits), progress_snapshot=snapshot,
                  progress_line_sha256=hashlib.sha256('\n'.join(lines).encode()).hexdigest())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=MODES, default='trace')
    a = p.parse_args()
    try:
        r = run(a.mode)
        rc = 0
    except Exception as exc:
        r = dict(verdict='FAIL', mode=a.mode, error_type=type(exc).__name__, error=str(exc),
                 traceback=traceback.format_exc())
        rc = 1
    r.update(mode=a.mode, schema='native41-progress-cpu/1', gpu_execution=False, box_evidence=False,
             probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps({k: r.get(k) for k in ('verdict', 'mode', 'error_type', 'error')}))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
