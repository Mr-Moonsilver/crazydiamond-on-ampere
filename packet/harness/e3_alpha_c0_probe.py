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

Fixture leaves: the parent's, unchanged (native41_progress_fixture_leaves.json), and stage_fixture_leaves.json for the suites.

Declared limit: no device stream and no real QSA layer on this path. What this adds over test_e3_stage_local is the real Core step,
real scheduler ticks, the real auto-arm cycle, native progress receipts and the wait path driving stage commands. The grade's join
runs over this run's real progress lines, so a stage arm's receipts are proven readable before the box.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
from unittest import mock

import e3_native41_progress_probe as pp
from stage_wire import StageScoped

MODES = ('local-burst', 'local-burst-solo', 'local-burst-skew', 'serial-burst', 'serial-burst-solo', 'serial-burst-skew')
ENV = {'FLASH_S5_NATIVE_QSA_POLICY': 'dense', 'FLASH_S5_STAGE_LOCAL': '1'}
CHUNK_METHODS = ('flash_s5_open', 'flash_s5_submit', 'flash_s5_poll', 'flash_s5_release', 'flash_s5_bind', 'flash_s5_receipt')


def run(mode):
    assert mode in MODES, mode
    local = mode.startswith('local')
    state = {}

    def connected(setting):
        import s5_startup as startup
        real = startup.install_engine

        def install(engine, *, env=None, **kwargs):
            env = dict(env or {}, FLASH_S5_NATIVE_PROGRESS=setting, **ENV)
            return real(engine, env=env, **kwargs)
        with mock.patch.object(startup, 'install_engine', install):
            conn = pp.ap.Connected('1')
        state['conn'] = conn
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

    with mock.patch.object(pp, 'connected', connected), mock.patch.object(pp, 'set_mode', set_mode):
        result = pp.run(mode.split('-', 1)[1])
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
    check('switch-installed-staged-driver', type(conn.driver).__name__ == 'StagedSegmentDriver',
          driver=type(conn.driver).__name__)
    check('native-chunks-ran', chunks > 0, chunks=chunks)
    if local:
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
