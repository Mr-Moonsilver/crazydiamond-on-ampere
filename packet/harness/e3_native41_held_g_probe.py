#!/usr/bin/env python3
"""Held-g connected CPU probe: the engine step anatomy (R5) on held-f's connected composition.

Runs e3_native41_progress_probe.run unchanged (production install_engine, real
EngineCore.step_with_batch_queue, scheduler, controller, driver, worker
bridges and dispatchers, the auto-arm policy, every held-f check) and also
captures the engine's anatomy lines. Per mode it checks:

* unset: no anatomy line;
* otherwise one engine anatomy flush per native chunk, with the chunk's
  identity and arm mode, no dropped row, known step classes only;
* the chunk's step sends equal its native commands (the 'n' rows of its
  progress line), so every send happened inside a recorded step;
* detail chunks (one in ANATOMY_DETAIL_EVERY) tile the chunk with
  contiguous steps; outside an owner wait a step sends at most one command
  (mode current: every step), and never more than one plus the wait cap;
* chunks begun with no live decoder (the solo modes) run empty stock steps,
  the pacing L3 removes; live chunks run stock steps with tokens.

L3 modes (``-wait``): the same held-f mode with the arm's idle selection
``wait`` (the native_progress control carries ``idle``). They add:

* the control reply and status carry idle ``wait``; every other mode ``step``
  with no idle step;
* chunks begun with no live decoder run an empty stock step only when the
  predicate refused for an input that step must carry (the aborted cohort's
  finished id), never a step with tokens; every other step is an idle step,
  the owner waiting for a native reply; live chunks have no idle step; the
  status idle_steps equals the anatomy's;
* at the first idle step, on the real scheduler and engine objects: the
  driver permit holds, and the stock-idle predicate holds and turns false
  for each single input a stock step would carry (running, waiting, skipped,
  finished or preempted ids, a new block to zero, a CoW copy, a freed
  encoder entry, a queued non-empty batch, a pending abort), and true again
  once each is restored; a queued empty batch keeps it true;
* the non-empty scheduled rows of the stock trace are reported as a hash,
  which the held-g self-run compares with the ``-step`` run of the same mode.

Fixture leaves are e3_native41_progress_probe's, plus:

* rpc-counter: its execute and sample fixtures do not go through
  collective_rpc, so the 'rpcs' column is zero here by construction; the
  stock RPC counter it reads is the one collective_rpc increments in
  test_e3_native_reply_hol.
* idle-wait-landed: DirectTransport computes every reply synchronously at
  submit, so a reply has always landed when the owner waits;
  ``wait_landed`` records the call and returns True. ReplyTransport's real
  wait is proven in test_e3_native_anatomy.IdleWaitTransport.
* idle-predicate-flips: the flips mutate the real scheduler containers by
  one sentinel entry each (or one queued batch entry, or an abort queue the
  CPU engine leaf otherwise lacks) and restore them before the step goes on.
"""
import argparse
import contextlib
import hashlib
import json
import logging
from pathlib import Path
import queue
import traceback
from types import SimpleNamespace as NS
from unittest import mock

import e3_native41_progress_probe as pp

WAIT_MODES = ('burst-solo-wait', 'current-solo-wait', 'burst-wait')
MODES = pp.MODES + WAIT_MODES
ANATOMY_LINE = 'flash-s5 anatomy '
PROGRESS_LINE = 'flash-s5 progress '


class Capture(pp.Capture):
    """Held-f's capture (progress lines in ``lines``) plus the anatomy lines."""
    anatomy = []
    progress = []

    def emit(self, record):
        super().emit(record)
        message = record.getMessage()
        if message.startswith(ANATOMY_LINE):
            Capture.anatomy.append(message)
        elif message.startswith(PROGRESS_LINE):
            Capture.progress.append(message)


def flushes(lines, prefix):
    out = {}
    for text in lines:
        body = json.loads(text[len(prefix):])
        out.setdefault((body['side'], body['line']), []).append(body)
    joined = []
    for key in sorted(out):
        parts = sorted(out[key], key=lambda b: b['part'])
        assert [b['part'] for b in parts] == list(range(parts[0]['parts'])), key
        rows = {}
        for b in parts:
            for tag, items in b['rows'].items():
                rows.setdefault(tag, []).extend(items)
        joined.append((parts[0], rows))
    return joined


def set_mode(conn, mode, idle):
    """pp.set_mode with the arm's idle selection in the native_progress control."""
    from vllm.v1.core.sched.interface import PauseState
    scheduler = conn.scheduler
    old = scheduler.pause_state
    scheduler.set_pause_state(PauseState.PAUSED_ALL)
    try:
        reply = conn.driver.submit_control(dict(action='native_progress', mode=mode, idle=idle,
                                                engine_epoch=conn.driver.engine_epoch))
        leased = conn.driver.auto_arm.manual_owned
        release = conn.driver.submit_control(dict(action='release_auto_arm',
                                                  engine_epoch=conn.driver.engine_epoch))
    finally:
        scheduler.set_pause_state(old)
    Idle.reply = reply
    return reply, leased, release


class Idle:
    """L3 observation: wait_landed leaf calls and the predicate flips at the first idle step."""
    reply = None
    landed = []
    flips = None
    conn = None
    # Permitted idle steps the predicate refused: what a stock step had to carry.
    refusals = []


def flip_checks(engine, hooks):
    """At a real idle state: each single stock input turns the predicate false, restored true."""
    scheduler = engine.scheduler
    sentinel = object()
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    out = dict(start=hooks._stock_step_idle(engine), permit=engine._flash_s5_driver.idle_wait_permit())

    def flip(name, add, remove):
        add()
        try:
            out[name] = hooks._stock_step_idle(engine)
        finally:
            remove()
        out[name + '-restored'] = hooks._stock_step_idle(engine)
    flip('running', lambda: scheduler.running.append(sentinel), scheduler.running.pop)
    flip('waiting', lambda: scheduler.waiting.append(sentinel), scheduler.waiting.pop)
    flip('skipped', lambda: scheduler.skipped_waiting.append(sentinel), scheduler.skipped_waiting.pop)
    flip('finished', lambda: scheduler.finished_req_ids.add('flip'),
         lambda: scheduler.finished_req_ids.discard('flip'))
    flip('preempted', lambda: scheduler.reset_preempted_req_ids.add('flip'),
         lambda: scheduler.reset_preempted_req_ids.discard('flip'))
    flip('new-block', lambda: managers[0].new_block_ids.append(1), managers[0].new_block_ids.pop)
    flip('cow-copy', lambda: managers[0]._pending_cow_copies.append((None, None)),
         managers[0]._pending_cow_copies.pop)
    flip('encoder-freed', lambda: scheduler.encoder_cache_manager.freed.append('flip'),
         scheduler.encoder_cache_manager.freed.pop)
    flip('queued-batch', lambda: engine.batch_queue.appendleft((None, NS(total_num_scheduled_tokens=3), None)),
         engine.batch_queue.popleft)
    aborts = queue.Queue()
    aborts.put(['flip'])

    def add_abort():
        engine.aborts_queue = aborts

    def remove_abort():
        del engine.aborts_queue
    flip('abort', add_abort, remove_abort)
    engine.batch_queue.appendleft((None, NS(total_num_scheduled_tokens=0), None))
    try:
        out['queued-empty-batch'] = hooks._stock_step_idle(engine)
    finally:
        engine.batch_queue.popleft()
    return out


def run(mode):
    Capture.anatomy, Capture.progress = [], []
    Idle.reply, Idle.landed, Idle.flips, Idle.conn, Idle.refusals = None, [], None, None, []
    wait = mode in WAIT_MODES
    base = mode[:-len('-wait')] if wait else mode
    real_connected = pp.connected
    restore = []

    def connected(setting):
        # pp.run has aliased the overlay modules by now: this is the mounted executor_hooks.
        import executor_hooks as hooks
        real_idle = hooks._stock_step_idle

        def stock_step_idle(engine):
            answer = real_idle(engine)
            driver = engine._flash_s5_driver
            if answer and Idle.flips is None and driver.idle_wait_permit():
                Idle.flips = {}
                Idle.flips = flip_checks(engine, NS(_stock_step_idle=real_idle))
            if not answer and driver.idle_wait_permit():
                s = engine.scheduler
                Idle.refusals.append(dict(
                    txn=driver.tx.txn_seq, finished=sorted(s.finished_req_ids),
                    preempted=len(s.reset_preempted_req_ids), running=len(s.running),
                    waiting=len(s.waiting) + len(s.skipped_waiting),
                    queued=[e[1].total_num_scheduled_tokens for e in engine.batch_queue]))
            return answer
        hooks._stock_step_idle = stock_step_idle
        restore.append((hooks, real_idle))
        conn = real_connected(setting)
        Idle.conn = conn

        def wait_landed(timeout_s):
            Idle.landed.append(timeout_s)
            return True
        conn.transport.wait_landed = wait_landed
        return conn
    patches = [mock.patch.object(pp, 'Capture', Capture), mock.patch.object(pp, 'connected', connected)]
    if wait:
        patches.append(mock.patch.object(pp, 'set_mode', lambda conn, m: set_mode(conn, m, 'wait')))
    try:
        with contextlib.ExitStack() as scope:
            for patch in patches:
                scope.enter_context(patch)
            result = pp.run(base)
    finally:
        for hooks, real_idle in restore:
            hooks._stock_step_idle = real_idle
    import native_measurement as nm
    import engine_driver as ed
    checks = result['checks']

    def check(name, ok, **details):
        checks.append(dict(name=name, ok=bool(ok), **details))
        assert ok, name + ' ' + json.dumps(details, sort_keys=True, default=str)[:2000]
    lines = Capture.anatomy
    if mode == 'unset':
        check('unset-no-anatomy', not lines)
        result.update(anatomy_lines=0)
        return result
    check('anatomy-lines-bounded', lines and all(len(t) < nm.PROGRESS_LINE_BYTES for t in lines))
    anatomy = flushes(lines, ANATOMY_LINE)
    check('only-engine-anatomy-in-process', all(h['side'] == 'engine' for h, _ in anatomy))
    # The progress lines, re-read from the same records the held-f checks used.
    progress = flushes(Capture.progress, PROGRESS_LINE)
    native = result['native']
    check('one-anatomy-flush-per-native-chunk', len(anatomy) == len(native)
          and [(h['chunk_start'], h['chunk_end']) for h, _ in anatomy]
          == [(r['start'], r['end']) for r in native], flushes=len(anatomy), chunks=len(native))
    check('anatomy-fields-and-no-drop', all(
        h['fields'] == {k: list(v) for k, v in nm.ANATOMY_ROWS.items()} and h['dropped'] == 0
        and h['schema'] == nm.ANATOMY_SCHEMA for h, _ in anatomy))
    fields = nm.ANATOMY_ROWS
    limit = ed.NativeProgress.WAIT_SEND_LIMIT
    burst = base in ('burst', 'burst-skew', 'burst-solo')
    summary = []
    for index, (head, rows) in enumerate(anatomy):
        e = [dict(zip(fields['e'], r)) for r in rows.get('e', [])]
        p = [dict(zip(fields['p'], r)) for r in rows.get('p', [])]
        classes = {r['cls']: r['steps'] for r in e}
        sends = sum(r['sends'] for r in e)
        txn = head['txn']
        check('chunk-%d-anatomy-identity' % txn, head['mode'] == ('burst' if burst else 'current')
              and type(head['decode_live']) is int and head['detail'] == int(index % nm.ANATOMY_DETAIL_EVERY == 0),
              head={k: head.get(k) for k in ('txn', 'mode', 'decode_live', 'detail')})
        check('chunk-%d-known-classes' % txn, classes and set(classes) <= set(nm.StepAnatomy.CLASSES),
              classes=classes)
        commands = None
        if index < len(progress):
            ph, prow = progress[index]
            commands = len(prow.get('n', []))
            check('chunk-%d-progress-and-anatomy-are-the-same-chunk' % txn,
                  (ph['txn'], ph['chunk_start'], ph['chunk_end']) == (txn, head['chunk_start'], head['chunk_end']))
        check('chunk-%d-sends-are-the-native-commands' % txn, sends == commands, sends=sends, commands=commands)
        if p:
            check('chunk-%d-detail-steps-tile-the-chunk' % txn, len(p) == sum(classes.values()) and all(
                abs(a[1] + a[2] - b[1]) <= 1 for a, b in
                zip([(r['cls'], r['t_begin'], r['total_us']) for r in p][:-1],
                    [(r['cls'], r['t_begin'], r['total_us']) for r in p][1:])),
                steps=len(p))
            cap = limit + 1 if burst else 1
            check('chunk-%d-sends-per-step-bounded' % txn, all(r['sends'] <= cap for r in p),
                  most=max(r['sends'] for r in p), cap=cap)
        if head['decode_live'] == 0:
            if not wait:
                check('chunk-%d-solo-paces-on-empty-stock-steps' % txn, classes.get('empty', 0) > 0
                      and classes.get('stock', 0) == 0, classes=classes)
        else:
            check('chunk-%d-live-runs-stock-steps' % txn, classes.get('stock', 0) > 0, classes=classes)
        summary.append(dict(txn=txn, decode_live=head['decode_live'], classes=classes, sends=sends,
                            total_us={r['cls']: r['total_us'] for r in e}))
    snapshot = result['progress_snapshot']
    idle_steps = sum(s['classes'].get('idle', 0) for s in summary)
    check('status-idle-steps-are-the-anatomy-idle-steps', snapshot['idle_steps'] == idle_steps
          and snapshot['idle_timeouts'] == 0 and len(Idle.landed) == idle_steps
          and all(v == ed.NativeProgress.IDLE_WAIT_S for v in Idle.landed),
          snapshot_idle=snapshot['idle_steps'], anatomy_idle=idle_steps, landed=len(Idle.landed))
    check('arm-idle-selection', snapshot['idle'] == ('wait' if wait else 'step')
          and all(head['idle'] == snapshot['idle'] for head, _ in anatomy),
          idle=snapshot['idle'])
    if wait:
        check('idle-control-reply', Idle.reply is not None and Idle.reply['idle'] == 'wait'
              and Idle.reply['previous_idle'] == 'step', reply={k: (Idle.reply or {}).get(k)
                                                                 for k in ('idle', 'previous_idle', 'mode')})
    for s in summary:
        if not wait:
            check('chunk-%d-no-idle-step-with-step-pacing' % s['txn'], 'idle' not in s['classes'])
        elif s['decode_live'] == 0:
            # A stock step runs only for something it must carry (here the
            # aborted cohort's finished id); every other step is an idle step.
            mine = [r for r in Idle.refusals if r['txn'] == s['txn']]
            carried = all(r['finished'] or r['preempted'] or r['running'] or r['waiting']
                          or any(r['queued']) for r in mine)
            check('chunk-%d-solo-runs-stock-steps-only-to-carry-stock-input' % s['txn'],
                  s['classes'].get('idle', 0) > 0 and not s['classes'].get('stock')
                  and s['classes'].get('empty', 0) == len(mine) and carried,
                  classes=s['classes'], refusals=mine)
        else:
            check('chunk-%d-live-runs-no-idle-step' % s['txn'], 'idle' not in s['classes'],
                  classes=s['classes'])
    solo = base.endswith('-solo')
    if wait and solo:
        flips = Idle.flips or {}
        expected = dict(start=True, permit=True, **{'queued-empty-batch': True})
        for name in ('running', 'waiting', 'skipped', 'finished', 'preempted', 'new-block', 'cow-copy',
                     'encoder-freed', 'queued-batch', 'abort'):
            expected[name] = False
            expected[name + '-restored'] = True
        check('idle-predicate-flips-on-the-real-scheduler', flips == expected, flips=flips)
    else:
        check('no-idle-state-without-a-solo-wait-arm', Idle.flips is None or wait,
              flips=Idle.flips)
    rows = [row['scheduled'] for row in result.get('trace', ()) if row['scheduled']]
    result.update(anatomy_lines=len(lines), anatomy_flushes=len(anatomy), anatomy=summary,
                  idle_flips=Idle.flips, idle_landed=len(Idle.landed), idle_refusals=Idle.refusals,
                  scheduled_nonempty=len(rows),
                  scheduled_nonempty_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                  anatomy_line_sha256=hashlib.sha256('\n'.join(lines).encode()).hexdigest())
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
    r.update(mode=a.mode, schema='native41-held-g-cpu/1', gpu_execution=False, box_evidence=False,
             probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps({k: r.get(k) for k in ('verdict', 'mode', 'error_type', 'error')}))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
