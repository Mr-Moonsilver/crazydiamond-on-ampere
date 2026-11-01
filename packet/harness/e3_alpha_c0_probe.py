#!/usr/bin/env python3
"""C0 connected CPU probe: stage-local chunks through the real generated executor path.

e3_native41_progress_probe.run, called unchanged (held-f's composition and all its checks: production install_engine, real
EngineCore.step_with_batch_queue, scheduler, controller, driver, worker bridges, dispatchers and the auto-arm policy), with
FLASH_S5_NATIVE_PROGRESS=burst, FLASH_S5_NATIVE_QSA_POLICY=dense and FLASH_S5_STAGE_LOCAL=1 passed to install_engine through its
env argument, and the arm's control sequence with the typed selection. Modes:

  local-burst, local-burst-solo, local-burst-skew      selection true
  serial-burst, serial-burst-solo, serial-burst-skew   the switch installed, selection false

The -solo and -skew forms are the parent's. The serial modes must reproduce alpha-perf's frozen progress-burst* records byte for
byte (the self-run's comparison): installing the switch moves nothing when the arm selects serial.

Fixture leaves: the parent's, unchanged (native41_progress_fixture_leaves.json), and stage_fixture_leaves.json for the suites. The
parent's CPU-worker driver subclass sits under the real StagedSegmentDriver: it adds workers after install and overrides nothing else.
stage_progress.ring_has_headroom answers true here because this composition's transport has no ring.

Declared limit: no device stream and no real QSA layer on this path. What this adds over test_e3_stage_local is the real Core step,
real scheduler ticks, the real auto-arm cycle, engine progress lines and the wait path driving stage commands. This composition has no
worker log lines; the grade's join over real engine and worker lines is test_alpha_c0_queue.RealLines. On a local arm the parent's
text runs with the WAIVED checks recorded instead of fatal, and each has a stand-in below.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback
import types
from types import SimpleNamespace
from unittest import mock

import e3_native41_progress_probe as pp

MODES = ('local-burst', 'local-burst-solo', 'local-burst-skew', 'serial-burst', 'serial-burst-solo', 'serial-burst-skew')
ENV = {'FLASH_S5_NATIVE_QSA_POLICY': 'dense', 'FLASH_S5_STAGE_LOCAL': '1'}
CHUNK_METHODS = ('flash_s5_open', 'flash_s5_submit', 'flash_s5_poll', 'flash_s5_release', 'flash_s5_bind', 'flash_s5_receipt')


# Parent checks with the serial path's shape, waived on a local arm only. Each is recorded in the result with its details, and the
# stage-path check that stands in for it is named here and run after the parent's run.
WAIVED = {
    r'chunk-\d+-command-flow': 'stage-chunk-command-flow: the row after the last release can be the sender receipt or a poll',
    r'wait-permit-granted-only-in-its-states': 'stage-wait-sends: the parent reads one phase; a staged chunk has one per stage',
    r'tick-sampled-permit-true-only-in-its-states-and-false-at-every-tick-state': 'stage-wait-sends: as above',
}
PARENT_ASSERT = "        assert ok, name + ' ' + json.dumps(details, sort_keys=True, default=str)[:2000]\n"
PARENT_RESULT = "    result.update(progress_lines=len(lines), progress_rows=len(seqs),"


def local_parent():
    """The parent's module text with two declared deltas: a waived check is recorded, not fatal; the engine lines stay in the result."""
    text = Path(pp.__file__).read_text()
    assert text.count(PARENT_ASSERT) == 2 and text.count(PARENT_RESULT) == 1  # run and _run_solo each have the check
    text = text.replace(PARENT_ASSERT, "        assert ok or ALPHA_C0_WAIVED(name, details), name + ' ' + json.dumps(details, sort_keys=True, default=str)[:2000]\n")
    text = text.replace(PARENT_RESULT, "    result.update(alpha_c0_lines=list(lines))\n" + PARENT_RESULT)
    module = types.ModuleType('e3_alpha_c0_local_parent')
    module.__file__ = pp.__file__
    waived = []

    def is_waived(name, details):
        hit = any(re.fullmatch(pattern, name) for pattern in WAIVED)
        if hit:
            waived.append(dict(name=name, details=details))
        return hit
    module.ALPHA_C0_WAIVED = is_waived
    sys.modules[module.__name__] = module
    exec(compile(text, pp.__file__, 'exec'), module.__dict__)
    return module, waived


def stage_checks(check, lines, wait_segments):
    """Stand-ins for the waived checks, over the real engine lines of this run."""
    import native_measurement as nm
    at = nm.PROGRESS_N
    bodies = [json.loads(t[len(nm.PROGRESS_LINE):]) for t in lines]
    chunks = {}
    for body in bodies:
        chunks.setdefault(body['line'], []).append(body)
    for number in sorted(chunks):
        parts = sorted(chunks[number], key=lambda b: b['part'])
        rows = [r for b in parts for r in b['rows'].get('n', [])]
        kinds = [r[at['kind']] for r in rows]
        last = max(i for i, k in enumerate(kinds) if k == 'release') if 'release' in kinds else -1
        check('stage-chunk-command-flow-%d' % number, kinds and kinds.count('open') == 2 and kinds[0] == 'open'
              and kinds.count('release') == 2 and 'segment' in kinds and set(kinds[last + 1:]) <= {'receipt', 'poll'},
              kinds=sorted(set(kinds)), after_release=kinds[last + 1:])
        waits = [r for r in rows if r[at['origin']] == 'wait']
        check('stage-wait-sends-%d' % number, all(
            (r[at['kind']] == 'segment' and r[at['segment']] in wait_segments) or r[at['kind']] == 'profile' for r in waits),
            kinds=sorted({(r[at['kind']], r[at['segment']]) for r in waits}, key=str))


def scoped_submit(transport):
    """test_e3_r4_worker_bridge.DirectTransport.submit with one addition: a StageScoped record reads as the box's ring does.

    Every rank still gets the command, through the real eh.worker_rpc; a non-target answers STAGE_SKIP and is not expected. Everything
    else keeps DirectTransport's rule, including its refusal of a plain broadcast expecting fewer ranks. Leaf scoped-direct-transport.
    """
    from vllm.v1.flash_e3.stage_wire import STAGE_SKIP, StageScoped

    def submit(kind, send, ranks=None, timeout_s=300.0, aggregate=None, precheck=None):
        if precheck is not None:
            precheck()
        sequence = transport.next
        transport.next += 1
        transport._calls[sequence] = SimpleNamespace(ready=False)
        send(sequence)
        method, args, kwargs, output_rank = transport.sender.envelopes[-1]
        everyone = tuple(sorted(transport.network.workers))
        scoped = bool(args) and type(args[0]) is StageScoped
        if scoped and output_rank is None:
            assert tuple(ranks) == args[0].ranks, 'a stage command expects its own ranks'
            targets = args[0].ranks
        elif output_rank is None:
            assert tuple(ranks) == everyone, 'untargeted broadcast must expect every rank'
            targets = everyone
        else:
            assert tuple(ranks) == (output_rank,), 'targeted reply registered the wrong ranks'
            targets = (output_rank,)
        values, failure = [], None
        for rank in everyone:
            if rank in transport.missing:
                continue
            try:
                value = getattr(transport.network.workers[rank], method)(*args, **kwargs)
            except BaseException as exc:
                if rank not in targets:
                    transport.silent_worker_errors.append((method, rank, type(exc).__name__))
                elif failure is None:
                    failure = exc
                continue
            if scoped:
                assert (value is STAGE_SKIP) == (rank not in args[0].ranks), 'a stage skip follows the target ranks'
            if rank in targets:
                values.append(value)
        if failure is not None:
            raise failure
        transport.ready[sequence] = values
        transport._calls[sequence].ready = not transport.hold
        return sequence
    return submit


def run(mode):
    assert mode in MODES, mode
    local = mode.startswith('local')
    state = {}

    def connected(setting):
        import s5_startup as startup
        real = startup.install_engine

        def install(engine, *, env=None, **kwargs):
            env = dict(env or {}, FLASH_S5_NATIVE_PROGRESS=setting, **ENV)
            # The parent's fixture adds its CPU workers in a NativeSegmentDriver subclass, patched in for this install. The
            # switch selects the staged class, so that subclass goes under it, unchanged. Real startup still does the choosing.
            import vllm.v1.flash_e3.engine_driver as ed
            import vllm.v1.flash_e3.stage_progress as sp

            class StagedWithCPUWorkers(sp.StagedSegmentDriver, ed.NativeSegmentDriver):
                pass
            # Leaf: this composition's transport has no broadcast ring, so the reserve is declared free here. The ring rule on a
            # real slot layout is test_e3_stage_local's, and the box reports headroom_refused.
            sp.ring_has_headroom = lambda sender, slots: True
            with mock.patch.object(sp, 'StagedSegmentDriver', StagedWithCPUWorkers):
                return real(engine, env=env, **kwargs)
        with mock.patch.object(startup, 'install_engine', install):
            conn = pp.ap.Connected('1')
        state['conn'] = conn
        conn.transport.submit = scoped_submit(conn.transport)
        return conn

    def set_mode(conn, selected):
        """The box arm's control sequence with the selection: pause, native_progress, native_stage_local, release, resume."""
        from vllm.v1.core.sched.interface import PauseState
        scheduler = conn.scheduler
        old = scheduler.pause_state
        scheduler.set_pause_state(PauseState.PAUSED_ALL)
        epoch = conn.driver.engine_epoch
        try:
            reply = conn.driver.submit_control(dict(action='native_progress', mode=selected, engine_epoch=epoch))
            state['selection'] = conn.driver.submit_control(dict(action='native_stage_local', local=local, engine_epoch=epoch))
            leased = conn.driver.auto_arm.manual_owned
            release = conn.driver.submit_control(dict(action='release_auto_arm', engine_epoch=epoch))
        finally:
            scheduler.set_pause_state(old)
        return reply, leased, release

    parent, waived = local_parent() if local else (pp, [])
    with mock.patch.object(parent, 'connected', connected), mock.patch.object(parent, 'set_mode', set_mode):
        result = parent.run(mode.split('-', 1)[1])
    # After the parent's overlay aliases: the class the installed driver wraps its commands in.
    from vllm.v1.flash_e3.stage_wire import StageScoped
    conn = state['conn']
    checks = result['checks']

    def check(name, ok, **details):
        checks.append(dict(name=name, ok=bool(ok), **details))
        assert ok, name + ' ' + json.dumps(details, sort_keys=True, default=str)[:2000]
    chunks = len(conn.native)
    status = conn.driver._status_record()['native_stage_local']
    counters = status['counters']
    selection = state['selection']
    check('selection-through-the-arm-sequence', selection['local'] is local and selection['route'] == 'full',
          reply={k: selection[k] for k in ('local', 'previous', 'route')})
    chunk_envelopes = [e for e in conn.sender.envelopes if e[0] in CHUNK_METHODS]
    scoped = [e for e in chunk_envelopes if e[1] and type(e[1][0]) is StageScoped]
    check('switch-installed-staged-driver', type(conn.driver).__name__ == 'StagedWithCPUWorkers',
          driver=type(conn.driver).__name__)
    check('native-chunks-ran', chunks > 0, chunks=chunks)
    if local:
        stage_checks(check, result.pop('alpha_c0_lines'), parent.WAIT_SEGMENTS)
        result['waived_parent_checks'] = sorted({row['name'] for row in waived})
        result['waiver_reasons'] = WAIVED
        check('stage-counters', status['selected'] is True and counters['chunks'] == chunks
              and counters['local_releases'] == 2 * chunks, counters=counters, chunks=chunks)
        check('every-chunk-command-names-its-stage', chunk_envelopes and len(scoped) == len(chunk_envelopes),
              unscoped=sorted({e[0] for e in chunk_envelopes if e not in scoped}))
        check('each-rank-answers-only-its-stage', all(set(e[1][0].ranks) in ({0, 1}, {2, 3}) or len(e[1][0].ranks) == 1
                                                      for e in scoped))
    else:
        check('serial-counters-all-zero', status['selected'] is False and not any(
            counters[k] for k in ('chunks', 'local_releases', 'both_stages_ns', 'headroom_refused')), counters=counters)
        check('no-chunk-command-is-scoped', not scoped, scoped=len(scoped))
    result.update(native_stage_local=counters, selection=selection.get('local'), chunk_commands=len(chunk_envelopes))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    a = p.parse_args()
    try:
        r = run(a.mode)
        rc = 0
    except Exception as exc:
        r = dict(verdict='FAIL', mode=a.mode, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        rc = 1
    r.update(mode=a.mode, schema='alpha-c0-connected-cpu/1', gpu_execution=False, box_evidence=False,
             probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=2, default=str)
        f.write('\n')
    print(json.dumps({k: r.get(k) for k in ('verdict', 'mode', 'error_type', 'error')}))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
