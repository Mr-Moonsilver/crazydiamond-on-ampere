#!/usr/bin/env python3
"""Load-robust twin of test_e3_w1_decode_driver.DriverTest.test_streams_advance_out_of_order_but_all_must_open.

The frozen test staggers each stream's first event with a sleep (0.04, 0.0, 0.02 s) measured from that
stream's own request arrival, so under load thread start skew and reader wake-ups larger than 20 ms flip
the expected advance order.  Here the order is made by the test, not by the clock:

* the fake server's ``_measure`` sends its headers, then waits on a per-stream gate (an Event) instead of
  sleeping; no gate is opened before all three measure requests have arrived;
* the test opens gate 1, waits on the driver's own ``state.opened`` for stream 1 (set by the reader
  together with ``first_at_ns``), then opens gate 2, waits, then opens gate 0;
* the driver's states are read through ``TrackedStreamState``, installed over ``D.StreamState`` with
  ``mock.patch.object``; it keeps the constructor and fields and only records the instance.

Every wait has a 30 s liveness deadline; a miss raises ``HarnessDeadline`` (an ERROR classified HARNESS,
never a product verdict).  The assertions are the frozen test's: rc 0, ``advance_order == [1, 2, 0]``,
``opened_streams == 3``.  Fixture parameters (open_timeout_s above the summed deadlines, budget, window,
read and drain bounds) and the leaves are declared in alpha_next_robust_fixture_leaves.json.  The frozen
module supplies the fake server, config and controller; none of its tests are loaded from here.
"""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_e3_w1_decode_driver as frozen  # noqa: E402 - fixtures only (module object, so no test is re-collected)

D = frozen.D
LIVENESS_S = 30.0
GATE_ORDER = (1, 2, 0)
STREAMS = len(GATE_ORDER)
# Fixture parameters of the fake run (not product thresholds): the open deadline covers the four
# summed liveness waits (all requests arrived, then three gated opens); the budget and window keep the
# post-open phase off the scheduler (a 4 ms stream must advance once inside 0.5 s and must not finish
# before it); read and drain bounds sit at or above the liveness deadline.
OPEN_TIMEOUT_S = 4 * LIVENESS_S + 30.0
GENERATION_BUDGET = 2000
WINDOW_S = 0.5
ENV = {'READ_TIMEOUT_S': str(2 * LIVENESS_S), 'DRAIN_TIMEOUT_S': str(LIVENESS_S)}


class HarnessDeadline(Exception):
    """A liveness deadline of the fixture expired: HARNESS, not a product verdict."""

    def __init__(self, what):
        super().__init__('HARNESS: liveness deadline (%.0f s) missed: %s' % (LIVENESS_S, what))


class TrackedStreamState(D.StreamState):
    """Fixture leaf: the driver's StreamState with the same constructor and fields, recorded for the test."""

    registry = None  # replaced per test with mock.patch.object

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        TrackedStreamState.registry.append(self)


class GatedConfig(frozen.FakeConfig):
    """The frozen FakeConfig without start delays, plus one gate per stream and an all-arrived event."""

    def __init__(self, streams=STREAMS, **overrides):
        super().__init__(streams=streams, **overrides)
        self.gates = {index: threading.Event() for index in range(streams)}
        self.arrived = set()
        self.arrived_lock = threading.Lock()
        self.all_arrived = threading.Event()
        self.gate_misses = []

    def open_all(self):
        self.all_arrived.set()
        for gate in self.gates.values():
            gate.set()


class _GatedHandler(frozen._Handler):
    def _measure(self, cfg, stream, budget):
        if not self._headers():
            return
        with cfg.arrived_lock:
            cfg.arrived.add(stream)
            if len(cfg.arrived) >= cfg.streams:
                cfg.all_arrived.set()
        gate = cfg.gates.get(stream)
        if gate is not None and not (cfg.all_arrived.wait(LIVENESS_S) and gate.wait(LIVENESS_S)):
            cfg.gate_misses.append(stream)
            return
        self._headers = lambda: not self._gone()  # already sent above; the frozen body resends otherwise
        super()._measure(cfg, stream, budget)


class GatedServer(frozen.FakeServer):
    """frozen.FakeServer with the gated handler; record/measure_requests/stop are the frozen ones."""

    def __init__(self, config):
        self.config = config
        self.requests = []
        self.lock = threading.Lock()
        self.httpd = frozen._CohortHTTPServer(('127.0.0.1', 0), _GatedHandler)
        self.httpd.fake = self
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:%d' % self.httpd.server_address[1]


def _wait(event, what):
    if not event.wait(LIVENESS_S):
        raise HarnessDeadline(what)


class OrderedOpenTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def _manifest(self):
        manifest = frozen._deep_update(json.loads(json.dumps(frozen.BASE_MANIFEST)), {
            'streams': STREAMS, 'repeats': 1, 'route_plan': ['split'], 'open_timeout_s': OPEN_TIMEOUT_S,
            'generation_budget': GENERATION_BUDGET, 'window_s': WINDOW_S})
        path = Path(self.temp.name) / 'manifest-ordered.json'
        path.write_text(json.dumps(manifest))
        return path, manifest

    def test_streams_advance_out_of_order_but_all_must_open(self):
        config = GatedConfig()
        server = GatedServer(config)
        self.addCleanup(server.stop)
        self.addCleanup(config.open_all)  # never leave a handler parked on a gate
        manifest_path, manifest = self._manifest()
        out_path = Path(self.temp.name) / 'out-ordered.json'
        env = dict({'WORKLOAD': str(manifest_path), 'RUN_ID': 'test-run', 'CONTROLLER_TIMEOUT_S': '3'}, **ENV)
        controller = frozen.FakeController(streams=manifest['streams'])
        run = dict(rc=None, error=None)
        done = threading.Event()

        def drive():
            try:
                run['rc'] = D.main([server.url, 'model-x', str(out_path)], controller=controller, env=env)
            except BaseException as exc:  # noqa: BLE001 - reported below
                run['error'] = exc
            finally:
                done.set()

        with mock.patch.object(TrackedStreamState, 'registry', []), \
                mock.patch.object(D, 'StreamState', TrackedStreamState):
            driver = threading.Thread(target=drive, daemon=True, name='ordered-driver')
            driver.start()
            _wait(config.all_arrived, 'all %d measure requests at the fake server' % STREAMS)
            states = {state.index: state for state in TrackedStreamState.registry}
            self.assertEqual(sorted(states), list(range(STREAMS)), 'one tracked state per stream')
            for index in GATE_ORDER:
                config.gates[index].set()
                _wait(states[index].opened, 'driver state.opened for stream %d' % index)
            _wait(done, 'driver main returned')
        driver.join(LIVENESS_S)
        self.assertEqual(config.gate_misses, [])
        if run['error'] is not None:
            raise run['error']
        rc = run['rc']
        if not out_path.exists():
            self.fail('driver did not write ' + str(out_path))
        data = json.loads(out_path.read_text())
        self.assertEqual(rc, 0, data['summary'])
        order = data['repeats'][0]['open']['advance_order']
        self.assertEqual(order, [1, 2, 0])
        self.assertEqual(data['repeats'][0]['open']['opened_streams'], 3)


if __name__ == '__main__':
    unittest.main()
