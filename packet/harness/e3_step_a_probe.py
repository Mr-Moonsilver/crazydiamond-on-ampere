#!/usr/bin/env python3
"""Step A connected CPU probe: a fused chunk through the real generated executor path.

e3_native41_progress_probe.run, called unchanged (held-f's composition and all its checks: production install_engine, real
EngineCore.step_with_batch_queue, scheduler, controller, driver, worker bridges, dispatchers and the auto-arm policy), with
FLASH_S5_NATIVE_PROGRESS=burst, FLASH_S5_NATIVE_QSA_POLICY=dense and FLASH_S5_QSA_FUSED=1 passed to install_engine through its
env argument, and the arm's control sequence with the typed selection. Modes:

  fused-burst, fused-burst-solo, fused-burst-skew      selection true
  stepped-burst, stepped-burst-solo, stepped-burst-skew the switch installed, selection false

The -solo and -skew forms are the parent's. The stepped modes must reproduce alpha-perf's frozen progress-burst* records byte
for byte (the self-run's comparison): installing the switch moves nothing when the arm selects stepped.

Fixture leaves, beyond the parent's (step_a_fixture_leaves.json):

* connected-layer-driver: LeafQsaDriver for the factory's FakeDriver. The dispatcher's fused path calls its fused_profiles and
  arm_fused. The real QsaLayerDriver and QsaPrefillOperation under a fused grant are not on this path: they run in
  test_e3_native_qsa_fused (FusedNetwork) and in the V3 device probe. test_e3_native_qsa_fused also proves the real fused_shape
  equals the engine's fused_expected_shape, which this leaf returns.
* connected-tp-leaf: the real PrefillTpCommunicator and CollectiveLedger over the declared LeafPynccl, entered from FakeLane.layer
  once per layer with one fixed tensor, so both peers report a real ledger for the real driver to agree.

Declared limit: no device stream, no real QSA layer on this path. What this adds over FusedNetwork is the real Core step, real
scheduler ticks, the real auto-arm cycle, native progress receipts and the wait path driving fused commands.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
from types import SimpleNamespace as NS
from unittest import mock

import e3_native41_progress_probe as pp

MODES = ('fused-burst', 'fused-burst-solo', 'fused-burst-skew', 'stepped-burst', 'stepped-burst-solo', 'stepped-burst-skew')
ENV = {'FLASH_S5_NATIVE_QSA_POLICY': 'dense', 'FLASH_S5_QSA_FUSED': '1'}
LAYERS_PER_STAGE = 2       # the parent's FakeLane: (0, 1) and (2, 3)
STAGES = 2


class LeafPynccl:
    """Leaf connected-tp-leaf: the PyNccl object under the real communicator."""
    disabled = False
    world_size = 2
    unique_id = NS(internal=b'tp-lane')   # what the ledger names a communicator by

    def __init__(self, rank):
        self.rank = rank
        self.calls = 0

    def all_reduce(self, tensor, stream=None):
        self.calls += 1
        return tensor


class LeafTensor:
    """Leaf connected-tp-leaf: what the ledger reads of a message."""
    shape = (1024, 8)
    dtype = 'float16'


class LeafQsaDriver:
    """Leaf connected-layer-driver (see the module docstring); the parent's FakeDriver's surface plus the two fused calls."""

    def __init__(self, shape_of):
        self.shape_of = shape_of
        self.armed = []
        self.aborted = None

    def cursor(self):
        return ('qsa', 1, 0)

    def abort(self, reason):
        self.aborted = reason
        return reason

    def fused_profiles(self, lane, valid_tokens):
        return tuple((layer, self.shape_of(valid_tokens)) for layer in lane.layers)

    def arm_fused(self, layer, grant):
        assert grant.query.layer == layer and grant.fused is True
        self.armed.append(layer)


def install_leaves(conn, ed, lane_comm):
    conn.tp_leaves, conn.leaf_drivers = {}, {}
    torch = NS(cuda=NS(current_stream=lambda device: 'lane'))
    for rank, worker in conn.workers.items():
        leaf = LeafPynccl(rank % 2)
        comm = lane_comm.PrefillTpCommunicator(leaf, 'cuda:0', torch_module=torch)
        worker.dispatcher.native.comms.tp = comm
        conn.tp_leaves[rank] = (leaf, comm)

        def factory(operation, module, context, rank=rank):
            driver = LeafQsaDriver(ed.fused_expected_shape)
            conn.leaf_drivers.setdefault(rank, []).append(driver)
            return driver
        worker.dispatcher.driver_factory = factory
        real_layer = worker.lane.layer

        def layer(index, comm=comm, real_layer=real_layer):
            comm.all_reduce(LeafTensor())
            return real_layer(index)
        worker.lane.layer = layer


def run(mode):
    assert mode in MODES, mode
    fused = mode.startswith('fused')
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
        """The box arm's control sequence with the selection: pause, native_progress, native_qsa_fused, release, resume."""
        from vllm.v1.core.sched.interface import PauseState
        scheduler = conn.scheduler
        old = scheduler.pause_state
        scheduler.set_pause_state(PauseState.PAUSED_ALL)
        epoch = conn.driver.engine_epoch
        try:
            reply = conn.driver.submit_control(dict(action='native_progress', mode=selected, engine_epoch=epoch))
            state['selection'] = conn.driver.submit_control(dict(action='native_qsa_fused', fused=fused, engine_epoch=epoch))
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
    counters = conn.driver._status_record()['native_qsa_fused']
    layers = STAGES * LAYERS_PER_STAGE * chunks
    selection = state['selection']
    check('selection-through-the-arm-sequence', selection['fused'] is fused and selection['previous'] is False
          and selection['route'] == 'full', reply={k: selection[k] for k in ('fused', 'previous', 'route')})
    opens = [args[0] for method, args, _, _ in conn.sender.envelopes if method == 'flash_s5_open']
    check('every-open-carries-the-latched-selection', opens and {o.qsa_fused for o in opens} == {fused})
    armed = {rank: [x for d in drivers for x in d.armed] for rank, drivers in conn.leaf_drivers.items()}
    collectives = {rank: (leaf.calls, comm.collectives) for rank, (leaf, comm) in conn.tp_leaves.items()}
    check('no-ledger-after-the-last-release', all(comm.ledger is None for _, comm in conn.tp_leaves.values()))
    check('every-stage-peer-entered-its-layers', all(c[0] == c[1] == LAYERS_PER_STAGE * chunks for c in collectives.values()),
          collectives=collectives)
    if fused:
        check('fused-counters', counters['selected'] is True and counters['chunks'] == chunks and counters['selections'] == 1
              and counters['fused_commands'] == layers and counters['grants'] == layers
              and counters['ledger_checks'] >= layers, counters=counters, chunks=chunks)
        check('every-layer-armed-by-its-grant', all(len(rows) == LAYERS_PER_STAGE * chunks for rows in armed.values())
              and set(armed[0]) == set(armed[1]) == {0, 1} and set(armed[2]) == set(armed[3]) == {2, 3}, armed=armed)
    else:
        check('stepped-counters-all-zero', counters['selected'] is False and not any(
            counters[k] for k in ('chunks', 'selections', 'grants', 'fused_commands', 'run_ahead_polls', 'ledger_checks')),
            counters=counters)
        check('no-layer-armed', not any(armed.values()), armed=armed)
    result.update(native_qsa_fused=counters, selection=selection.get('fused'), layers_per_chunk=STAGES * LAYERS_PER_STAGE)
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
    r.update(mode=a.mode, schema='step-a-connected-cpu/1', gpu_execution=False, box_evidence=False,
             probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=2, default=str)
        f.write('\n')
    print(json.dumps({k: r.get(k) for k in ('verdict', 'mode', 'error_type', 'error')}))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
