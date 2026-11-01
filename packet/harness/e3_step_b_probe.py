#!/usr/bin/env python3
"""Step B connected CPU probe: a group chunk through the real generated executor path.

e3_native41_progress_probe.run, called unchanged (held-f's composition and all its checks: production install_engine, real
EngineCore.step_with_batch_queue, scheduler, controller, driver, worker bridges, dispatchers and the auto-arm policy), with
FLASH_S5_NATIVE_PROGRESS=burst, FLASH_S5_NATIVE_QSA_POLICY=dense, FLASH_S5_QSA_FUSED=1 and FLASH_S5_LAYER_GROUPS=1 passed to install_engine
through its env argument, and the arm's control sequence with both typed selections. Modes:

  g1-burst, g1-burst-solo, g1-burst-skew    size 1, the switch installed: every layer its own command
  g2-burst, g2-burst-solo, g2-burst-skew    size 2: each stage's two layers one command
  g4-burst                                  size 4: the same two layers, the stage end closing the group early

The -solo and -skew forms are the parent's. Each group mode must have the same request tokens and native chunks as its size 1 mode.

Fixture leaves, beyond Step A's probe (step_b_fixture_leaves.json):

* connected-group-driver: LeafGroupDriver for Step A's LeafQsaDriver. Step A's leaf grants a QSA profile on every layer, which would
  close every group at one layer, so this one grants only each stage's last layer. Real group composition over the real QSA driver is
  test_e3_native_layer_groups (GroupNetwork); real kernels are the VM's.
* connected-lane-end: FakeLane.end_layer, which the real dispatcher's group path reads and the parent's FakeLane does not have: the
  stage's last layer plus one, the same value as the real LaneForward's.

Declared limit: no device stream and no real QSA layer on this path. What this adds over GroupNetwork is the real Core step, real scheduler
ticks, the real auto-arm cycle, native progress receipts and the wait path driving group commands.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
from unittest import mock

import e3_step_a_probe as sp

pp = sp.pp
MODES = ('g1-burst', 'g1-burst-solo', 'g1-burst-skew', 'g2-burst', 'g2-burst-solo', 'g2-burst-skew', 'g4-burst')
ENV = dict(sp.ENV, FLASH_S5_LAYER_GROUPS='1')
LAYERS_PER_STAGE = sp.LAYERS_PER_STAGE
STAGES = sp.STAGES


class LeafGroupDriver(sp.LeafQsaDriver):
    """Leaf connected-group-driver (see the module docstring)."""

    def fused_profiles(self, lane, valid_tokens):
        return ((max(lane.layers), self.shape_of(valid_tokens)),)


def install_leaves(conn, ed, lane_comm):
    with mock.patch.object(sp, 'LeafQsaDriver', LeafGroupDriver):
        sp.install_leaves(conn, ed, lane_comm)
    for rank, worker in conn.workers.items():
        # Leaf connected-lane-end.
        worker.lane.end_layer = 2 if rank < 2 else 4
        original = worker.dispatcher.driver_factory

        def factory(operation, module, context, original=original, rank=rank):
            with mock.patch.object(sp, 'LeafQsaDriver', LeafGroupDriver):
                return original(operation, module, context)
        worker.dispatcher.driver_factory = factory


def run(mode):
    assert mode in MODES, mode
    size = int(mode[1])
    state = {}

    def connected(setting):
        import s5_startup as startup
        import engine_driver as ed
        import lane_comm
        real = startup.install_engine

        def install(engine, *, env=None, **kwargs):
            env = dict(env or {}, FLASH_S5_NATIVE_PROGRESS=setting, **ENV)
            return real(engine, env=env, **kwargs)
        with mock.patch.object(startup, 'install_engine', install):
            conn = pp.ap.Connected('1')
        install_leaves(conn, ed, lane_comm)
        state['conn'] = conn
        return conn

    def set_mode(conn, selected):
        """The box arm's control sequence: pause, native_progress, native_qsa_fused, native_layer_group, release, resume."""
        from vllm.v1.core.sched.interface import PauseState
        scheduler = conn.scheduler
        old = scheduler.pause_state
        scheduler.set_pause_state(PauseState.PAUSED_ALL)
        epoch = conn.driver.engine_epoch
        try:
            reply = conn.driver.submit_control(dict(action='native_progress', mode=selected, engine_epoch=epoch))
            state['fused'] = conn.driver.submit_control(dict(action='native_qsa_fused', fused=True, engine_epoch=epoch))
            state['group'] = conn.driver.submit_control(dict(action='native_layer_group', size=size, engine_epoch=epoch))
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
    status = conn.driver._status_record()
    fused, groups = status['native_qsa_fused'], status['native_layer_groups']
    layers = STAGES * LAYERS_PER_STAGE * chunks
    check('selections-through-the-arm-sequence', state['fused']['fused'] is True and state['group']['size'] == size
          and state['group']['previous'] == 1 and state['group']['route'] == 'full', reply={k: state['group'][k] for k in ('size', 'previous', 'route')})
    envelopes = conn.sender.envelopes
    opens = [args[0] for method, args, _, _ in envelopes if method == 'flash_s5_open']
    check('every-open-carries-the-latched-size', opens and {o.layer_group for o in opens} == {size} and {o.qsa_fused for o in opens} == {True})
    submitted = [args[0] for method, args, _, _ in envelopes if method == 'flash_s5_submit']
    layer_commands = {(c.transaction, c.descriptor.stage, c.descriptor.layer_start, c.descriptor.layer_end) for c in submitted
                      if c.descriptor.layers}
    want = STAGES * chunks if size != 1 else layers
    check('commands-per-chunk-follow-the-size', len(layer_commands) == want, commands=len(layer_commands), want=want)
    check('every-command-names-its-size', {c.descriptor.layer_group for c in submitted if c.descriptor.layers} == {size})
    check('no-ledger-after-the-last-release', all(comm.ledger is None for _, comm in conn.tp_leaves.values()))
    collectives = {rank: (leaf.calls, comm.collectives) for rank, (leaf, comm) in conn.tp_leaves.items()}
    # The leaf enters one collective per layer, whatever the command holds: the device work per layer does not move with the size.
    check('every-stage-peer-entered-every-layer', all(c[0] == c[1] == LAYERS_PER_STAGE * chunks for c in collectives.values()),
          collectives=collectives)
    armed = {rank: [x for d in drivers for x in d.armed] for rank, drivers in conn.leaf_drivers.items()}
    check('only-the-granted-layer-armed', all(len(rows) == chunks for rows in armed.values())
          and set(armed[0]) == set(armed[1]) == {1} and set(armed[2]) == set(armed[3]) == {3}, armed=armed)
    check('fused-counters', fused['selected'] is True and fused['chunks'] == chunks and fused['fused_commands'] == STAGES * chunks
          and fused['grants'] == STAGES * chunks, counters=fused)
    if size == 1:
        check('size-one-group-counters-all-zero', not any(groups[k] for k in ('chunks', 'plan_commands', 'group_commands', 'group_layers', 'selections')),
              counters=groups)
    else:
        check('group-counters', groups['size'] == size and groups['selections'] == 1 and groups['chunks'] == chunks
              and groups['plan_commands'] == groups['group_commands'] == STAGES * chunks and groups['group_layers'] == layers
              and groups['group_layers_max'] == LAYERS_PER_STAGE, counters=groups)
    result.update(native_qsa_fused=fused, native_layer_groups=groups, size=size, layers_per_chunk=STAGES * LAYERS_PER_STAGE)
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
    r.update(mode=a.mode, schema='step-b-connected-cpu/1', gpu_execution=False, box_evidence=False,
             probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=2, default=str)
        f.write('\n')
    print(json.dumps({k: r.get(k) for k in ('verdict', 'mode', 'error_type', 'error')}))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
