#!/usr/bin/env python3
"""Held-f connected CPU probe: native progress receipts over a real automatic cycle.

Same composition as e3_native41_auto_probe (production install_engine, real
EngineCore.step_with_batch_queue, scheduler, controller, driver, worker bridges
and dispatchers). The setting reaches install_engine through its env argument.
Fixture leaves (native41_fixture_leaves.json plus progress-meta-pop):

* progress-meta-pop: DirectTransport calls each worker method directly. The
  generated ``WorkerProc._execute_worker_rpc`` pops the reserved kwarg through
  ``executor_hooks.worker_rpc_begin`` before dispatch; this leaf calls that same
  real hook on a per-rank copy of the kwargs, so a worker method never sees it.
* DirectTransport carries no ``registered_s``/lag and the ordinary execute is the
  probe's fixture, not ``collective_rpc``: per-rank receive times and stock rows
  ('s'/'w') are therefore absent here by construction. They are proven over the
  generated worker loop and the real ReplyTransport in test_e3_native_reply_hol.
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

MODES = ('unset', 'trace', 'trace-skew')


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


def run(mode):
    # Overlay modules are aliased first, exactly as e3_native41_auto_probe.run.
    root, failures = overlay_aliases()
    import executor_hooks as eh
    import native_measurement as nm
    setting = None if mode == 'unset' else 'trace'
    conn = connected(setting)
    conn.dispatched = set()
    popped = []
    conn.transport.network = NS(workers={rank: PopMeta(w, rank, eh, popped)
                                         for rank, w in conn.workers.items()})
    capture = Capture()
    import engine_driver as ed
    # The modules' own loggers (vllm.v1.flash_e3.* under the overlay aliases).
    loggers = [ed._progress_logger(), eh._progress_logger()]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(capture)
        lg.setLevel(logging.INFO)
    try:
        with contextlib.ExitStack() as scope:
            if mode == 'trace-skew':
                conn.skew = scope.enter_context(ap.Skew(conn).installed())
            result = ap._run('tp-skew' if mode == 'trace-skew' else 'auto', conn, root, failures)
    finally:
        for lg, level in zip(loggers, levels):
            lg.removeHandler(capture)
            lg.setLevel(level)
    checks = result['checks']

    def check(name, ok, **details):
        checks.append(dict(name=name, ok=bool(ok), **details))
        assert ok, name
    progress = conn.driver.progress
    if setting is None:
        check('unset-no-progress', progress is None and not capture.lines and not popped
              and getattr(conn.executor, '_flash_s5_progress_recording', None) is None)
        result.update(progress_lines=0)
        return result
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
              and parts[0]['dropped'] == 0 and parts[0]['fields'] == {k: list(v) for k, v in nm.PROGRESS_ROWS.items()}
              and all('fields' not in b for b in parts[1:]))
        rows = [r for b in parts for r in b['rows'].get('n', [])]
        joined.append((parts[0], rows))
    native = conn.native
    check('one-flush-per-native-chunk', len(joined) == len(native)
          and [(h['chunk_start'], h['chunk_end']) for h, _ in joined]
          == [(r['start'], r['end']) for r in native],
          flushes=len(joined), chunks=len(native))
    field = {name: i for i, name in enumerate(nm.PROGRESS_ROWS['n'])}
    seqs = []
    for head, rows in joined:
        kinds = [r[field['kind']] for r in rows]
        seqs += [r[field['seq']] for r in rows]
        check('chunk-%d-command-flow' % head['txn'],
              kinds and kinds[0] == 'open' and kinds[-1] == 'release' and 'segment' in kinds
              and head['open_rows'] == 0 and head['mode'] == 'current'
              and type(head['decode_live']) is int and head['decode_live'] >= 1,
              kinds=sorted(set(kinds)), decode_live=head['decode_live'])
        check('chunk-%d-stamps-ordered' % head['txn'], all(
            r[field['t_submit']] <= r[field['t_sent']] <= r[field['t_apply']]
            and r[field['direct']] == 0 and r[field['origin']] == 'tick'
            and r[field['outcome']] is not None for r in rows))
        segments = [r for r in rows if r[field['kind']] == 'segment']
        check('chunk-%d-segment-identity' % head['txn'], all(
            type(r[field['op_seq']]) is int and r[field['stage']] in (0, 1) for r in segments))
    sent = sorted({seq for rank, seq, name, direct in popped if rank == 0})
    check('every-recorded-command-carried-meta-and-every-meta-was-recorded',
          sorted(seqs) == sent and len(set(seqs)) == len(seqs)
          and all(not direct for _rank, _seq, _name, direct in popped)
          and {rank for rank, *_ in popped} == set(conn.workers),
          recorded=len(seqs), sent=len(sent))
    snapshot = conn.driver._status_record()['native_progress']
    check('status-receipt', snapshot['installed'] == 'trace' and snapshot['mode'] == 'current'
          and snapshot['chunks'] == len(native) and snapshot['line_errors'] == 0
          and snapshot['commands'] == len(seqs) and snapshot['direct_commands'] == 0
          and snapshot['trace_dropped'] == 0 and not snapshot['recording'], snapshot=snapshot)
    check('not-recording-after-cycle', getattr(conn.executor, '_flash_s5_progress_recording', None) is None)
    result.update(progress_lines=len(lines), progress_rows=len(seqs),
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
